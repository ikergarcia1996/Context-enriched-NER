import subprocess


def run_bash_command(command: str) -> None:
    print(command)
    subprocess.run(["bash", "-c", command])


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(input_path: str) -> int:
    with open(input_path, "r", encoding="utf8") as f:
        return sum(bl.count("\n") for bl in blocks(f))
