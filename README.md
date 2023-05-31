# [lpr](https://github.com/dudushy/lpr)
License Plate Recognition.

---

<!-- ## Prerequisites:
- [Python 3.11.3](https://www.python.org/downloads/release/python-3113/) -->

## How to install:
- Install Python and Tesseract:
    ```bash
    sudo apt update && sudo apt upgrade
    sudo apt install python3.11 && sudo apt install python3-pip
    sudo apt-get install tesseract-ocr
    ```

- Add Python to PATH:
    ```bash
    nano ~/.bashrc
    ```

    ```bash
    PYTHONPATH=$HOME/lib/python

    export PYTHONPATH
    ```

- Install Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## How to run:
```bash
python3 src/main.py
```
