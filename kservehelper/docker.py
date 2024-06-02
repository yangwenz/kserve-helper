import os
import yaml
import logging
import subprocess
from queue import Queue
from threading import Thread
from typing import Dict, Sequence
from schema import Schema, Optional, SchemaError

logging.basicConfig(level="INFO")
logger = logging.getLogger("Docker build")

SCHEMA = Schema(
    {
        "build": {
            Optional("cuda"): lambda x: x in ["11.7", "11.8", "12.1", "12.2"],
            Optional("python_version"): lambda x: x in ["3.8", "3.9", "3.10"],
            Optional("cuda_image_type"): lambda x: x in ["base", "runtime", "devel"],
            Optional("cudnn"): bool,
            Optional("ngc"): bool,
            Optional("commands"): list,
            Optional("system_packages"): list,
            Optional("python_requirements"): list,
            Optional("python_packages"): list,
            Optional("additional_commands"): list,
        },
        "image": str,
        "entrypoint": str
    }
)

DOCKER_IMAGES = {
    "default": "ubuntu22.04",
    "python:3.8": "python:3.8.18-slim",
    "python:3.9": "python:3.9.18-slim",
    "python:3.10": "python:3.10.13-slim",
    # https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
    "cuda:11.7,python:3.8": "nvidia/cuda:11.7.1-base-ubuntu20.04",
    "cuda:11.8,python:3.8": "nvidia/cuda:11.8.0-base-ubuntu20.04",
    "cuda:12.1,python:3.8": "nvidia/cuda:12.1.0-base-ubuntu20.04",
    "cuda:12.2,python:3.8": "nvidia/cuda:12.2.2-base-ubuntu20.04",
    "cuda:11.7,python:3.10": "nvidia/cuda:11.7.1-base-ubuntu22.04",
    "cuda:11.8,python:3.10": "nvidia/cuda:11.8.0-base-ubuntu22.04",
    "cuda:12.1,python:3.10": "nvidia/cuda:12.1.0-base-ubuntu22.04",
    "cuda:12.2,python:3.10": "nvidia/cuda:12.2.2-base-ubuntu22.04",
}
CONFIGFILE = "config.yaml"
DOCKERFILE = "Dockerfile"

IGNORED_FILES = [
    "__pycache__",
    CONFIGFILE,
    DOCKERFILE
]


def _get_cuda_image(cuda_version: str, python_version: str, image_type: str, cudnn: bool):
    assert image_type in ["base", "runtime", "devel"]
    cudas = {
        "11.7": "11.7.1",
        "11.8": "11.8.0",
        "12.1": "12.1.0",
        "12.2": "12.2.2"
    }
    pythons = {
        "3.8": "ubuntu20.04",
        "3.10": "ubuntu22.04"
    }
    parts = [f"nvidia/cuda:{cudas.get(cuda_version, cuda_version)}"]
    if cudnn:
        parts.append("cudnn8")
    parts.append(image_type)
    parts.append(pythons.get(python_version, "ubuntu22.04"))
    return "-".join(parts)


def _get_ngc_image(cuda_version: str, python_version: str):
    images = {
        "12.2": {
            "3.10": "nvcr.io/nvidia/pytorch:23.08-py3"
        },
        "12.1": {
            "3.8": "nvcr.io/nvidia/pytorch:23.04-py3",
            "3.10": "nvcr.io/nvidia/pytorch:23.07-py3"
        },
        "11.8": {
            "3.8": "nvcr.io/nvidia/pytorch:22.12-py3",
        },
        "11.7": {
            "3.8": "nvcr.io/nvidia/pytorch:22.08-py3",
        }
    }
    assert cuda_version in images, \
        f"Choose cuda version from {images.keys()}"
    assert python_version in images[cuda_version], \
        f"Choose python version from {images[cuda_version].keys()}"
    return images[cuda_version][python_version]


def _load_config(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    try:
        SCHEMA.validate(config)
        return config
    except SchemaError as e:
        raise RuntimeError(
            "The image config does not fit the required schema."
        ) from e


def _write_dockerfile(config: Dict, folder: str):
    # Choose the base image
    python_installed = False
    if "cuda" in config["build"]:
        if not config["build"].get("ngc", False):
            base_image = _get_cuda_image(
                cuda_version=config["build"]["cuda"],
                python_version=config["build"].get("python_version", "3.10"),
                image_type=config["build"].get("cuda_image_type", "base"),
                cudnn=config["build"].get("cudnn", False),
            )
        else:
            base_image = _get_ngc_image(
                cuda_version=config["build"]["cuda"],
                python_version=config["build"].get("python_version", "3.10"),
            )
            python_installed = True
    elif "python_version" in config["build"]:
        tag = f"python:{config['build']['python_version']}"
        base_image = DOCKER_IMAGES[tag]
        python_installed = True
    else:
        base_image = DOCKER_IMAGES["default"]

    commands = [
        f"FROM {base_image}",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "RUN apt update --fix-missing && apt upgrade -y",
        ""
    ]
    # Additional commands
    if "commands" in config["build"] and len(config["build"]["commands"]) > 0:
        for command in config["build"]["commands"]:
            commands.append(f"RUN {command}")
        commands.append("")

    # Install system packages
    if "system_packages" in config["build"] and len(config["build"]["system_packages"]) > 0:
        commands.append("RUN apt install -y --no-install-recommends \\")
        for package in config["build"]["system_packages"][:-1]:
            commands.append(f"    {package} \\")
        commands.append(f"    {config['build']['system_packages'][-1]}")
        commands.append("")

    # Install default python
    if not python_installed:
        commands.append("RUN apt install -y python3 python3-pip")
        commands.append("")

    # Install python requirements
    if "python_requirements" in config["build"]:
        for requirement_file in config["build"]["python_requirements"]:
            commands.append(f"COPY {requirement_file} .")
            commands.append(f"RUN pip install -r {requirement_file}")
            commands.append("")

    # Install python dependencies
    if "python_packages" in config["build"] and len(config["build"]["python_packages"]) > 0:
        # Copy a wheel package if exists
        for package in config["build"]["python_packages"]:
            if package.endswith(".whl"):
                commands.append(f"COPY {package} .")
        commands.append("")
        # Pip install packages
        commands.append("RUN pip install --no-cache-dir --upgrade pip")
        commands.append("RUN pip install --no-cache-dir \\")
        for package in config["build"]["python_packages"][:-1]:
            commands.append(f"    {package} \\")
        commands.append(f"    {config['build']['python_packages'][-1]}")
        commands.append("")

    # Extra commands
    if "additional_commands" in config["build"] and len(config["build"]["additional_commands"]) > 0:
        for command in config["build"]["additional_commands"]:
            commands.append(f"RUN {command}")
        commands.append("")

    # Copy the code and artifacts
    commands.append("WORKDIR /app")
    commands.append("")
    for file in os.listdir(folder):
        if file not in IGNORED_FILES:
            if not os.path.isdir(file):
                commands.append(f"COPY {file} .")
            else:
                commands.append(f"COPY {file}/ {file}/")
    commands.append("")
    commands.append(f'CMD ["python3", "/app/{config["entrypoint"]}"]')

    with open(os.path.join(folder, DOCKERFILE), "w") as f:
        for command in commands:
            f.write(f"{command}\n")


# Adapted from BentoML: https://github.com/bentoml/BentoML/blob/main/src/bentoml/_internal/container/base.py
def _stream_logs(cmds: Sequence[str], *, env=None, cwd=None):
    proc = subprocess.Popen(
        cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd
    )
    queue: Queue[tuple[str, bytes]] = Queue()
    stderr, stdout = b"", b""
    # We will use a thread to read from the subprocess and avoid hanging from Ctrl+C
    t = Thread(target=_enqueue_output, args=(proc.stdout, "stdout", queue))
    t.daemon = True
    t.start()
    t = Thread(target=_enqueue_output, args=(proc.stderr, "stderr", queue))
    t.daemon = True
    t.start()
    for _ in range(2):
        for src, line in iter(queue.get, None):
            logger.info(line.decode(errors="replace").strip("\n"))
            if src == "stderr":
                stderr += line
            else:
                stdout += line
    exit_code = proc.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(
            exit_code, cmds, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(
        proc.args, exit_code, stdout=stdout, stderr=stderr
    )


def _enqueue_output(pipe, pipe_name, queue):
    try:
        with pipe:
            for line in iter(pipe.readline, b""):
                queue.put((pipe_name, line))
    finally:
        queue.put(None)


def build(folder: str, quiet: bool = False):
    # Build Dockerfile
    config = _load_config(os.path.join(folder, CONFIGFILE))
    _write_dockerfile(config, folder)
    # Build docker image
    try:
        command = ["docker", "build", "--platform=linux/amd64"]
        if quiet:
            command.append("--quiet")
        command += ["-t", config["image"], folder]
        _stream_logs(command)
    except subprocess.CalledProcessError as e:
        print(e)


def push(folder: str):
    config = _load_config(os.path.join(folder, CONFIGFILE))
    try:
        command = ["docker", "push", config["image"]]
        _stream_logs(command)
    except subprocess.CalledProcessError as e:
        print(e)
