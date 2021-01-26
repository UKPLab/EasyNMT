import sentencepiece as spm
from omegaconf import OmegaConf
import ast
import os
import time
import copy
import torch
from fairseq import checkpoint_utils, distributed_utils, tasks, utils
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from collections import namedtuple
from typing import List
import logging

logger = logging.getLogger(__name__)


class Fairseq:
    def __init__(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(model_path, 'spm.model'))
        with open(os.path.join(model_path, 'config.yaml')) as fIn:
            self.conf = OmegaConf.load(fIn)

        #Update paths in config
        self.conf.task.fixed_dictionary = os.path.join(model_path, self.conf.task.fixed_dictionary)
        self.conf.task.path = os.path.join(model_path, self.conf.task.path)
        self.conf.common_eval.path = os.path.join(model_path, self.conf.common_eval.path)

        self._lang_pairs = set(self.conf.task.lang_pairs.split(","))

        #Load Model
        cfg = self.conf

        utils.import_user_module(cfg.common)

        # Setup task, e.g., translation
        task = tasks.setup_task(cfg.task)

        #Set the dict object for all available languages
        self.shared_dict = next(iter(task.data_manager.dicts.values()))
        for lang in task.data_manager.langs:
            task.data_manager.dicts[lang] = copy.deepcopy(self.shared_dict)

        # Load ensemble
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)
        logger.info("Loading model(s) from {}".format(cfg.common_eval.path))
        models, model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

        # Optimize ensemble for generation
        for model in models:
            if model is None:
                continue
            model.prepare_for_inference_(cfg)

        self.models = models
        self.max_positions = utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        )


        self.task = task
        self.generator = task.build_generator(self.models, cfg.generation)
        ##New: add symbols_to_strip_from_output from config to generator
        if 'symbols_to_strip_from_output' in cfg.generation:
            for tok_id in cfg.generation.symbols_to_strip_from_output:
                self.generator.symbols_to_strip_from_output.add(tok_id)


    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str, device: str, beam_size: int = 5, is_tokenized: bool = False):
        if source_lang == target_lang:
            return sentences

        if source_lang+'-'+target_lang not in self._lang_pairs:
            raise ValueError("Translation for the language combination {}-{} not supported".format(source_lang, target_lang))

        if not is_tokenized:
            sentences = [" ".join(self.tokenizer.EncodeAsPieces(sent)) for sent in sentences]

        for model in self.models:
            if model is not None:
                model.to(device)
        cfg = self.conf
        task = self.task
        generator = self.generator

        cfg.generation.beam = beam_size
        cfg.task.source_lang = source_lang
        cfg.task.target_lang = target_lang

        task.args.source_lang = source_lang
        task.args.target_lang = target_lang

        generator.beam_size = beam_size

        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(cfg.generation.replace_unk)

        output = []
        for batch in self.make_batches(sentences, cfg, task, self.max_positions):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens.to(device)
            src_lengths = batch.src_lengths.to(device)
            constraints = batch.constraints


            if constraints is not None:
                constraints = constraints.to(device)

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }

            translations = task.inference_step(
                generator, self.models, sample, constraints=constraints
            )

            list_constraints = [[] for _ in range(bsz)]
            if cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]

            results = []
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                        },
                    )
                )

            # sort output to match input order
            for src_id, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)

                _, hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=hypos[0]["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypos[0]["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=self.get_symbols_to_strip_from_output(generator),
                )

                output.append(hypo_str)

        return output


    @staticmethod
    def make_batches(lines, cfg, task, max_positions, batch_size=None, max_tokens=None):
        Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")

        if cfg.generation.constraints:
            # Strip (tab-delimited) contraints, if present, from input lines,
            # store them in batch_constraints
            batch_constraints = [list() for _ in lines]
            for i, line in enumerate(lines):
                if "\t" in line:
                    lines[i], *batch_constraints[i] = line.split("\t")

            # Convert each List[str] to List[Tensor]
            for i, constraint_list in enumerate(batch_constraints):
                batch_constraints[i] = [
                    task.target_dictionary.encode_line(
                        constraint,
                        append_eos=False,
                        add_if_not_exist=False,
                    )
                    for constraint in constraint_list
                ]

        tokens = [
            task.source_dictionary.encode_line(
                src_str, add_if_not_exist=False
            ).long()
            for src_str in lines
        ]

        if cfg.generation.constraints:
            constraints_tensor = pack_constraints(batch_constraints)
        else:
            constraints_tensor = None

        lengths = [t.numel() for t in tokens]
        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(
                tokens, lengths, constraints=constraints_tensor
            ),
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            constraints = batch.get("constraints", None)

            yield Batch(
                ids=ids,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                constraints=constraints,
            )

    @staticmethod
    def get_symbols_to_strip_from_output(generator):
        if hasattr(generator, "symbols_to_strip_from_output"):
            return generator.symbols_to_strip_from_output
        else:
            return {generator.eos}