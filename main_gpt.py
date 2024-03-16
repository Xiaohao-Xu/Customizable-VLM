import os
import json
import argparse

import numpy as np

import utils
from PIL import Image
from alpaca_eval.decoders import openai as openai_decoder
from openai import OpenAI
import textwrap
import cv2
import base64
import requests
from tqdm import tqdm

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def load_dataset(path):
    with open(path, "r") as f:
        datasets = json.load(f)
    processed_dataset = []
    datasets_bar = tqdm(datasets)
    for data in datasets_bar:
        datasets_bar.set_description("Processing Dataset ")
        defect_img = encode_image(data['defect_img_path'])
        good_img_path = data['good_img_path']
        ref_num = 1
        number_range = range(0, ref_num)  
        good_imgs = []
        for number in number_range:
            good_img_path_ = good_img_path.replace("000", f"{number:03}")

            if not os.path.isfile(good_img_path_):
                break
            good_img = encode_image(good_img_path_)
            good_imgs.append(good_img)

        object_type = data['object_type']
        processed_dataset.append([data, good_imgs, defect_img])
    return processed_dataset

def load_cache(path):
    print("in cache")
    with open(path, "r") as f:
        return json.load(f)
    
def get_cache(cache,example,args):
    query = {}
    uncached_data = []
    if len(cache) == 0:
        return example
    for data in cache:
        query[data['defect_img_path']] = 1
    for c in example:
        if c[0]['defect_img_path'] not in query:
            uncached_data.append(c)
    return uncached_data

def clean_control_chars(text):
    # first eacape \
    text = text.replace("\\", "\\\\")

    # then escape \n, \t, \r
    text = text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return text

def process_batch(batch, args):
    from IPython import embed

    text_batch, good_imgs_batch, defect_img_batch = zip(*batch)
    processed_dataset = []
    new_batch = []
    prompts = []
    gpt_responses = []
    model_name = args.model
    new_text_batch = []
    for data_text, good_imgs, defect_img in zip(text_batch, good_imgs_batch, defect_img_batch):
        prompt = utils.generate_prompt(
            data_text['object_type'], args.prompt_template)
        prompts.append(prompt)
        input = []
        input.extend(good_imgs)
        input.append(defect_img)
        input.append(prompt)
        client = OpenAI(api_key=args.openai_api_key)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{good_imgs[0]}",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{defect_img}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            gpt_responses.append(response.choices[0].message.content)
        except:
            print("process error")
            continue
        new_text_batch.append(data_text)

    for i, (data, response) in enumerate(zip(new_text_batch, gpt_responses)):
        response = clean_control_chars(response)
        if args.verbose:
            print("response", response)
        try:
            response = json.loads(response)
            output = data.copy()
            output['reasoning'] = response['reasoning']
            output['correctness'] = response['correctness']
            processed_dataset.append(output)
        except:
            print("error in json parsing at index:", i, " Skipping this example")
            if args.debug:
                print(response)
                print(gpt_responses[i])
            new_batch.append(batch[i])
    return new_batch, processed_dataset


def main(args):
    dataset = load_dataset(args.dataset)
    print("dataset ready")
    if not os.path.exists(args.cache):
        cache = []
    else:
        cache = load_cache(args.cache)
    processed_dataset = []
    processed_dataset.extend(cache)
    bs = args.batch_size
    dataset = dataset[:int(len(dataset) * args.fraction)]
    batch_num = int(len(dataset) / bs) + 1
    unprocessed_batchs = []

    for i in tqdm(range(batch_num)):
        print(f"Processing {i+1}th batch_size")
        batch = dataset[i*bs:(i+1)*bs]
        batch = get_cache(cache, batch, args)
        if len(batch) == 0:
            continue
        print("batch len: ", len(batch))
        unprocessed_batch, batch_processed = process_batch(batch, args)
        processed_dataset.extend(batch_processed)
        unprocessed_batchs.extend(unprocessed_batch)
        if len(unprocessed_batch) == bs:
            print(" Entire batch failed, backing up the batch")
            batch = []

        if len(batch_processed) > 0:
            with open(args.output, "w") as f:
                json.dump(processed_dataset, f, indent=4)
    failure_count = 0
    while len(unprocessed_batchs) > 0 and failure_count < args.repeat_num:
        unprocessed_batchs, batch_processed = process_batch(unprocessed_batchs, args)
        processed_dataset.extend(batch_processed)
        with open(args.output, "w") as f:
            json.dump(processed_dataset, f, indent=4)
        failure_count += 1
    if failure_count == args.repeat_num:
        print("Failed to process the remaining batch")
        print("Remaining batch size", len(unprocessed_batchs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="datasets/")
    parser.add_argument("--cache",
                        type=str,
                        default="./output/answer.json")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview")
    parser.add_argument("--output",
                        type=str,
                        default="./output/answer.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--repeat_num", type=int, required=False, default=3)
    parser.add_argument("--fraction",
                        type=float,
                        required=False,
                        default=1)
    parser.add_argument("--prompt_template",
                        type=str,
                        default="./prompt_template/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--openai_api_key",
                        type=str,
                        default='')
    args = parser.parse_args()
    main(args)
