import argparse
import collections
import json
import time
import os

from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM
from shark_opt_wrapper import OPTForCausalLMModel

MODEL_NAME = "facebook/opt-1.3b"
MAX_SEQUENCE_LENGTH = 512
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


def create_vmfb_module(model_name, tokenizer, device):
    opt_base_model = OPTForCausalLM.from_pretrained(model_name)
    opt_base_model.eval()
    opt_model = OPTForCausalLMModel(opt_base_model)
    encoded_inputs = tokenizer(
        "What is the meaning of life?",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    inputs = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    # np.save("model_inputs_0.npy", inputs[0])
    # np.save("model_inputs_1.npy", inputs[1])

    opt_fs_name = get_opt_fs_name(model_name)
    mlir_path = f"./{opt_fs_name}_causallm_{MAX_SEQUENCE_LENGTH}_torch.mlir"
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

    vmfb_name = f"{opt_fs_name}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{DEVICE}_tiled_ukernels"
    shark_module.save_module(module_name=vmfb_name)
    vmfb_path = vmfb_name + ".vmfb"
    return vmfb_path


def load_shark_model(model_name) -> ModelWrapper:
    opt_fs_name = get_opt_fs_name(model_name)
    vmfb_name = f"{opt_fs_name}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{DEVICE}_tiled_ukernels.vmfb"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if not os.path.isfile(vmfb_name):
        print(f"vmfb not found. compiling and saving to {vmfb_name}")
        create_vmfb_module(MODEL_NAME, tokenizer, DEVICE)
    shark_module = SharkInference(mlir_module=None, device="cpu-task")
    shark_module.load_module(vmfb_name)
    return ModelWrapper(model=shark_module, tokenizer=tokenizer)


def run_shark_model(model_wrapper: ModelWrapper, tokens):
    # Generate logits output of OPT model.
    return model_wrapper.model("forward", tokens)


def load_huggingface_model() -> ModelWrapper:
    return ModelWrapper(
        model=OPTForCausalLM.from_pretrained(MODEL_NAME),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    )


def run_huggingface_model(model_wrapper: ModelWrapper, tokens):
    return model_wrapper.model.forward(tokens.input_ids,
                                       tokens.attention_mask,
                                       return_dict=False)


def run_huggingface():
    model_wrapper = load_huggingface_model()
    prompt = "What is the meaning of life?"
    logits = run_huggingface_model(model_wrapper, prompt)

    print(logits[0])


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)


def collect_huggingface_logits():
    t0 = time.time()
    model_wrapper = load_huggingface_model()
    print("--- Took {} seconds to load Huggingface.".format(time.time() - t0))
    results = []
    tokenized_prompts = []
    for prompt in PROMPTS:
        tokens = model_wrapper.tokenizer(
            prompt,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_prompts.append(tokens)
    t0 = time.time()
    for idx, tokens in enumerate(tokenized_prompts):
        print("prompt: {}".format(PROMPTS[idx]))
        logits = run_huggingface_model(model_wrapper, tokens)
        results.append([PROMPTS[idx], logits[0].tolist()])
    print("--- Took {} seconds to run Huggingface.".format(time.time() - t0))
    save_json(results, "/tmp/huggingface.json")


def collect_shark_logits(model_name):
    t0 = time.time()
    model_wrapper = load_shark_model(model_name)
    print("--- Took {} seconds to load Shark.".format(time.time() - t0))
    results = []
    tokenized_prompts = []
    for prompt in PROMPTS:
        tokens = model_wrapper.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
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
        results.append([PROMPTS[idx], lst])
    print("--- Took {} seconds to run Shark.".format(time.time() - t0))
    save_json(results, "/tmp/shark.json")


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
                        default='facebook/opt-1.3b')
    args = parser.parse_args()
    print('args={}'.format(args))
    return args


if __name__ == "__main__":
    parse_args()
    collect_shark_logits(MODEL_NAME)
    collect_huggingface_logits()
