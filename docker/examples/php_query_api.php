<?php
/* 
  This example shows how to query the EasyNMT Docker REST API (https://github.com/UKPLab/EasyNMT) using PHP
*/

// Here you define your REST API endpoit
$url = 'http://localhost:24080/translate';

// The data we want to transfer to the API
$data = [ 'text' => "Hallo Welt", 
          'target_lang' => "en"];

// We send the data as POST request using json encoded data
$options = [
	'http' => [
		'header'  => "Content-type: application/json",
		'method'  => 'POST',
		'content' => json_encode($data)
	]
];

// Sending the data to the server
$context = stream_context_create($options);
$result = file_get_contents($url, false, $context);

// We receive a json string from the server which we decode
$translated_data = json_decode($result);

var_dump($translated_data);