import argparse
import collections
import json
import time
from typing import Tuple
import os

from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM
from shark_opt_wrapper import OPTForCausalLMModel

DEVICE = "cpu"

PROMPTS = [
    "What is the meaning of life?",
    "Tell me something you don't know.",
    "What does Xilinx do?",
    "What is the mass of earth?",
    "What is a poem?",
    "What is recursion?",
    "Tell me a one line joke.",
    "Who is Gilgamesh?",
    "Tell me something about cryptocurrency.",
    "How did it all begin?",
]

ModelWrapper = collections.namedtuple("ModelWrapper", ["model", "tokenizer"])


def create_vmfb_module(model_name, tokenizer, device, max_seq_len):
    opt_base_model = OPTForCausalLM.from_pretrained(model_name)
    opt_base_model.eval()
    opt_model = OPTForCausalLMModel(opt_base_model)
    encoded_inputs = tokenizer(
        "What is the meaning of life?",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    inputs = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    # np.save("model_inputs_0.npy", inputs[0])
    # np.save("model_inputs_1.npy", inputs[1])

    opt_fs_name = get_opt_fs_name(model_name)
    mlir_path = f"./{opt_fs_name}_causallm_{max_seq_len}_torch.mlir"
    if os.path.isfile(mlir_path):
        with open(mlir_path, "r") as f:
            model_mlir = f.read()
        print(f"Loaded .mlir from {mlir_path}")
    else:
        (model_mlir, func_name) = import_with_fx(
            model=opt_model,
            inputs=inputs,
            is_f16=False,
            model_name=opt_fs_name,
            return_str=True,
        )
        with open(mlir_path, "w") as f:
            f.write(model_mlir)
        print(f"Saved mlir at {mlir_path}")

    shark_module = SharkInference(
        model_mlir,
        device=device,
        mlir_dialect="tm_tensor",
        is_benchmark=False,
    )

    vmfb_name = f"{opt_fs_name}_causallm_{max_seq_len}_torch_{DEVICE}_tiled_ukernels"
    shark_module.save_module(module_name=vmfb_name)
    vmfb_path = vmfb_name + ".vmfb"
    return vmfb_path


def load_shark_model(model_name: str, max_seq_len: int) -> ModelWrapper:
    opt_fs_name = get_opt_fs_name(model_name)
    vmfb_name = f"{opt_fs_name}_causallm_{max_seq_len}_torch_{DEVICE}_tiled_ukernels.vmfb"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if not os.path.isfile(vmfb_name):
        print(f"vmfb not found. compiling and saving to {vmfb_name}")
        create_vmfb_module(model_name, tokenizer, DEVICE, max_seq_len)
    shark_module = SharkInference(mlir_module=None, device="cpu-task")
    shark_module.load_module(vmfb_name)
    return ModelWrapper(model=shark_module, tokenizer=tokenizer)


def run_shark_model(model_wrapper: ModelWrapper, tokens):
    # Generate logits output of OPT model.
    return model_wrapper.model("forward", tokens)


def load_huggingface_model(model_name: str) -> ModelWrapper:
    return ModelWrapper(
        model=OPTForCausalLM.from_pretrained(model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
    )


def run_huggingface_model(model_wrapper: ModelWrapper, tokens):
    return model_wrapper.model.forward(tokens.input_ids,
                                       tokens.attention_mask,
                                       return_dict=False)


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)


def collect_huggingface_logits(model_name: str, max_seq_len: int,
                               save_json: bool) -> Tuple[float, float]:
    t0 = time.time()
    model_wrapper = load_huggingface_model(model_name)
    load_time = time.time() - t0
    print("--- Took {} seconds to load Huggingface.".format(load_time))
    results = []
    tokenized_prompts = []
    for prompt in PROMPTS:
        tokens = model_wrapper.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_prompts.append(tokens)
    t0 = time.time()
    for idx, tokens in enumerate(tokenized_prompts):
        print("prompt: {}".format(PROMPTS[idx]))
        logits = run_huggingface_model(model_wrapper, tokens)
        if save_json:
            results.append([PROMPTS[idx], logits[0].tolist()])
    run_time = time.time() - t0
    print("--- Took {} seconds to run Huggingface.".format(run_time))
    if save_json:
        save_json(results, "/tmp/huggingface.json")
    return (load_time, run_time)


def collect_shark_logits(model_name: str, max_seq_len: int,
                         save_json: bool) -> Tuple[float, float]:
    t0 = time.time()
    model_wrapper = load_shark_model(model_name, max_seq_len)
    load_time = time.time() - t0
    print("--- Took {} seconds to load Shark.".format(load_time))
    results = []
    tokenized_prompts = []
    for prompt in PROMPTS:
        tokens = model_wrapper.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        inputs = (
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        tokenized_prompts.append(inputs)
    t0 = time.time()
    for idx, tokens in enumerate(tokenized_prompts):
        print("prompt: {}".format(PROMPTS[idx]))
        logits = run_shark_model(model_wrapper, tokens)
        lst = [e.tolist() for e in logits]
        if save_json:
            results.append([PROMPTS[idx], lst])
    run_time = time.time() - t0
    print("--- Took {} seconds to run Shark.".format(run_time))
    if save_json:
        save_json(results, "/tmp/shark.json")
    return (load_time, run_time)


def get_opt_fs_name(model_name: str) -> str:
    """Cleanses the model name ino a file system-friendly name.

    Example: get_opt_fs_name('facebook/opt-1.3b') == 'opt_1-3b'
    """
    slash_split = model_name.split('/')
    assert 1 <= len(slash_split) <= 2, 'There should be at most one slash.'
    model_name = slash_split[-1]
    for src_pattern, dest_pattern in (('-', '_'), ('.', '-')):
        model_name = model_name.replace(src_pattern, dest_pattern)
    return model_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-json',
                        help='If set, saves output JSON.',
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--max-seq-len',
                        help='Max sequence length',
                        type=int,
                        default=32)
    parser.add_argument('--model-name',
                        help='Model name',
                        type=str,
                        choices=[
                            'facebook/opt-125m', 'facebook/opt-350m',
                            'facebook/opt-1.3b', 'facebook/opt-6.7b'
                        ],
                        default='facebook/opt-1.3b')
    args = parser.parse_args()
    print('args={}'.format(args))
    return args


if __name__ == "__main__":
    args = parse_args()
    shark_times = collect_shark_logits(args.model_name, args.max_seq_len,
                                       args.save_json)
    huggingface_times = collect_huggingface_logits(args.model_name,
                                                   args.max_seq_len,
                                                   args.save_json)
    # [model_name, max_seq_len, hark_load_time, shark_run_time,
    #  huggingface_load_time, huggingface_run_time]
    summary_fields = [args.model_name, args.max_seq_len
                      ] + list(shark_times) + list(huggingface_times)
    summary_json = json.dumps(summary_fields)
    print(f'# Summary: {summary_json}')
