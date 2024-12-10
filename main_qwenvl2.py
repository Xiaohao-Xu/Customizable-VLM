import os
import json
import argparse
import utils
from tqdm import tqdm
from openvlms.qwenvl2 import load_model, get_response

def load_dataset(path):
    with open(path, "r") as f:
        datasets = json.load(f)
    processed_dataset = []
    datasets_bar = tqdm(datasets)
    for data in datasets_bar:
        datasets_bar.set_description("Processing Dataset ")
        defect_img = data['defect_img_path']
        good_img_path = data['good_img_path']
        ref_num = 1
        number_range = range(0, ref_num)  
        good_imgs = []
        for number in number_range:
            good_img_path_ = good_img_path.replace("000", f"{number:03}")
            if not os.path.isfile(good_img_path_):
                break
            good_imgs.append(good_img_path_)

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

def process_batch(model, processor, batch, args):
    text_batch, good_imgs_batch, defect_img_batch = zip(*batch)
    processed_dataset = []
    new_batch = []
    prompts = []
    vlm_responses = []
    # model_name = args.model
    new_text_batch = []
    for data_text, good_imgs, defect_img in zip(text_batch, good_imgs_batch, defect_img_batch):
        prompt = utils.generate_prompt(
            data_text['object_type'], args.prompt_template)
        prompts.append(prompt)
        input = []
        input.extend(good_imgs)
        input.append(defect_img)
        input.append(prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": good_imgs[0]},
                    {"type": "image", "image": defect_img},
                    {"type": "text", "text": prompts},
                ],
            }
        ]
        output = get_response(model, processor, messages) # list, len(output) = len(messages)
        response = output[0]
        vlm_responses.append(response)
        new_text_batch.append(data_text)

    for i, (data, response) in enumerate(zip(new_text_batch, vlm_responses)):
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
                print(vlm_responses[i])
            new_batch.append(batch[i])
    return new_batch, processed_dataset


def main(args):
    model, processor = load_model(args.model, device='auto')
    print("model ready")
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
        unprocessed_batch, batch_processed = process_batch(model, processor, batch, args)
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
        unprocessed_batchs, batch_processed = process_batch(model, processor, unprocessed_batchs, args)
        processed_dataset.extend(batch_processed)
        with open(args.output, "w") as f:
            json.dump(processed_dataset, f, indent=4)
        failure_count += 1
    if failure_count == args.repeat_num:
        print("Failed to process the remaining batch")
        print("Remaining batch size", len(unprocessed_batchs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./ckpts/Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--dataset",
                        type=str,
                        default="datasets/MVTecAD/vlm_for_ad_dataset.json")
    parser.add_argument("--cache",
                        type=str,
                        default="./output/answer_qwenvl2_7b.json")
    parser.add_argument("--output",
                        type=str,
                        default="./output/answer_qwenvl2_7b.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fraction",
                        type=float,
                        required=False,
                        default=1)
    parser.add_argument("--prompt_template",
                        type=str,
                        default="./prompt_template/ad_prompt.txt")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
