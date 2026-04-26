import os
import sys
import subprocess

try:
    import pkg_resources
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "setuptools==69.5.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    import pkg_resources

import importlib.util

if importlib.util.find_spec("uv") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "uv"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


import pkg_resources

def install_prerequirements(file_path="prerequirements-versions.txt"):
    if not os.path.exists(file_path):
        return

    to_install = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "==" in line:
                pkg, required_version = line.split("==")
                try:
                    installed_version = pkg_resources.get_distribution(pkg).version
                    if installed_version != required_version:
                        to_install.append(line)
                except pkg_resources.DistributionNotFound:
                    to_install.append(line)
            else:
                to_install.append(line)

    if not to_install:
        return

    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(1)

    try:
        subprocess.run(["uv", "pip", "install", "--system", *to_install], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(1)

install_prerequirements()

import logging
import re
import subprocess
import os
import shutil
import sys
import importlib.util
import importlib.metadata
import platform
import json
import shlex
import socket
import urllib.request
from packaging import version
from functools import lru_cache
from typing import NamedTuple
from pathlib import Path

from modules import cmd_args, errors
from modules.paths_internal import script_path, extensions_dir, extensions_builtin_dir
from modules.timer import startup_timer
from modules import logging_config
from modules_forge import forge_version
from modules_forge.config import always_disabled_extensions


args, _ = cmd_args.parser.parse_known_args()
logging_config.setup_logging(args.loglevel)

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
dir_repos = "repositories"

default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')


def has_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


@lru_cache()
def commit_hash():
    try:
        return subprocess.check_output([git, "-C", script_path, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"


@lru_cache()
def git_tag_a1111():
    try:
        return subprocess.check_output([git, "-C", script_path, "describe", "--tags"], shell=False, encoding='utf8').strip()
    except Exception:
        try:

            changelog_md = os.path.join(script_path, "CHANGELOG.md")
            with open(changelog_md, "r", encoding="utf-8") as file:
                line = next((line.strip() for line in file if line.strip()), "<none>")
                line = line.replace("## ", "")
                return line
        except Exception:
            return "<none>"


def git_tag():
    return forge_version.version


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live):

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0


def git_fix_workspace(dir, name):
    run(f'"{git}" -C "{dir}" fetch --refetch --no-auto-gc', f"Fetching all contents for {name}", f"Couldn't fetch {name}", live=True)
    run(f'"{git}" -C "{dir}" gc --aggressive --prune=now', f"Pruning {name}", f"Couldn't prune {name}", live=True)
    return


def run_git(dir, name, command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live, autofix=True):
    try:
        return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)
    except RuntimeError:
        if not autofix:
            raise

    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(dir, name)

    return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)


def git_clone(url, dir, name, commithash=None):

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run_git(dir, name, 'rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        if current_hash == commithash:
            return

        if run_git(dir, name, 'config --get remote.origin.url', None, f"Couldn't determine {name}'s origin URL", live=False).strip() != url:
            run_git(dir, name, f'remote set-url origin "{url}"', None, f"Failed to set {name}'s origin URL", live=False)

        run_git(dir, name, 'fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}", autofix=False)

        run_git(dir, name, f'checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}", live=True)

        return

    try:
        run(f'"{git}" clone --config core.filemode=false "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)
    except RuntimeError:
        shutil.rmtree(dir, ignore_errors=True)
        raise

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        stdout = run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env).strip()
        if stdout:
            print(stdout)
    except Exception as e:
        errors.report(str(e))


def list_extensions(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n''', exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get('disabled_extensions', []) + always_disabled_extensions)
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none' or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def list_extensions_builtin(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n''', exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get('disabled_extensions', []))
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none' or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_builtin_dir):
        return []

    return [x for x in os.listdir(extensions_builtin_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(extensions_dir):
        return

    with startup_timer.subcategory("run extensions installers"):
        for dirname_extension in list_extensions(settings_file):
            logging.debug(f"Installing {dirname_extension}")

            path = os.path.join(extensions_dir, dirname_extension)

            if os.path.isdir(path):
                run_extension_installer(path)
                startup_timer.record(dirname_extension)

    if not os.path.isdir(extensions_builtin_dir):
        return

    with startup_timer.subcategory("run extensions_builtin installers"):
        for dirname_extension in list_extensions_builtin(settings_file):
            logging.debug(f"Installing {dirname_extension}")

            path = os.path.join(extensions_builtin_dir, dirname_extension)

            if os.path.isdir(path):
                run_extension_installer(path)
                startup_timer.record(dirname_extension)

    return


re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def requirements_met(requirements_file):

    import importlib.metadata
    import packaging.version

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "":
                continue

            m = re.match(re_requirement, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            if version_required == "":
                continue

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                return False

    return True


import os
os.environ["UV_LINK_MODE"] = "copy"

import subprocess

def prepare_environment():
    torch_command = [
        "uv", "pip", "install",
        "torch", "torchvision", "torchaudio", "xformers==0.0.31.post1",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
        "--system"
    ]
    triton_command = [
        "uv", "pip", "install", "triton-windows", "--system"
    ]

    subprocess.run(torch_command, check=True)

    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.31.post1')
    clip_package = os.environ.get('CLIP_PACKAGE', "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip")

    try:
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
    except OSError:
        pass

    commit = commit_hash()
    tag = git_tag()
    startup_timer.record("git version info")

    print(f"Python {sys.version}")
    print(f"Version: {tag}")
    print(f"Commit hash: {commit}")

    # CHECK INTERNET BEFORE ALL INSTALLS
    internet_available = has_internet()

    if not is_installed("clip"):
        if internet_available:
            run_pip(f"install {clip_package}", "clip")
            startup_timer.record("install clip")
        else:
            print("! No internet access. Skipping clip installation.")

    if not is_installed("open_clip"):
        if internet_available:
            run_pip(f"install {openclip_package}", "open_clip")
            startup_timer.record("install open_clip")
        else:
            print("! No internet access. Skipping open_clip installation.")

    if (not is_installed("xformers")):
        if internet_available:
            run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")
            startup_timer.record("install xformers")
        else:
            print("! No internet access. Skipping xformers installation.")

    if args.triton:
        if internet_available:
            subprocess.run(triton_command, check=True)
        else:
            print("! No internet access. Skipping triton installation.")

    os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

    startup_timer.record("clone repositores")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if not requirements_met(requirements_file):
        if internet_available:
            run_pip(f"install -r \"{requirements_file}\"", "requirements")
            startup_timer.record("install requirements")
        else:
            print("! No internet access. Requirements not met, but skipping installation.")

    run_extensions_installers(settings_file=args.ui_settings_file)

    ensure_numpy_version("1.26.4")

    ensure_protobuf_version("3.20.3")

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


def get_installed_numpy_version():
    try:
        if sys.version_info >= (3, 8):
            from importlib.metadata import version
        else:
            from pkg_resources import get_distribution as version
        return version("numpy")
    except Exception:
        return None


def ensure_numpy_version(target_version="1.26.4"):
    installed_version = get_installed_numpy_version()
    if installed_version == target_version:
        return

    print(f"Installing numpy=={target_version} (current: {installed_version})...")
    if has_internet():
        run(
            f"uv pip install --force-reinstall numpy=={target_version} --system",
            f"Installing numpy=={target_version}",
            "Couldn't install numpy",
            live=True
        )
    else:
        print("! No internet access. Skipping numpy update and continuing execution.")


def get_installed_protobuf_version():
    try:
        if sys.version_info >= (3, 8):
            from importlib.metadata import version
        else:
            from pkg_resources import get_distribution as version
        return version("protobuf")
    except Exception:
        return None


def ensure_protobuf_version(target_version="3.20.3"):
    installed_version = get_installed_protobuf_version()
    if installed_version == target_version:
        return

    print(f"Installing protobuf=={target_version} (current: {installed_version})...")
    if has_internet():
        run(
            f"uv pip install --force-reinstall protobuf=={target_version} --system",
            f"Installing protobuf=={target_version}",
            "Couldn't install protobuf",
            live=True
        )
    else:
        print("! No internet access. Skipping protobuf update and continuing execution.")


def configure_forge_reference_checkout(a1111_home: Path):
    class ModelRef(NamedTuple):
        arg_name: str
        relative_path: str

    refs = [
        ModelRef(arg_name="--ckpt-dir", relative_path="models/Stable-diffusion"),
        ModelRef(arg_name="--vae-dir", relative_path="models/VAE"),
        ModelRef(arg_name="--hypernetwork-dir", relative_path="models/hypernetworks"),
        ModelRef(arg_name="--embeddings-dir", relative_path="embeddings"),
        ModelRef(arg_name="--lora-dir", relative_path="models/lora"),
        ModelRef(arg_name="--controlnet-dir", relative_path="models/ControlNet"),
        ModelRef(arg_name="--controlnet-preprocessor-models-dir", relative_path="extensions/sd-webui-controlnet/annotator/downloads"),
    ]

    for ref in refs:
        target_path = a1111_home / ref.relative_path
        if not target_path.exists():
            print(f"Path {target_path} does not exist. Skip setting {ref.arg_name}")
            continue

        if ref.arg_name in sys.argv:
            continue

        sys.argv.append(ref.arg_name)
        sys.argv.append(str(target_path))


def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")
    import webui
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()

    from modules_forge import main_thread

    main_thread.loop()
    return


def dump_sysinfo():
    from modules import sysinfo
    import datetime

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename
