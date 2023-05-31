# pip freeze | xargs pip uninstall -y
# pip freeze > requirements.txt
# * -- Imports
import os
import glob
import sys

from lpr.lpr import LPR
from imutils import paths
import argparse
import imutils
import cv2


# * -- Variables
title = "LPR"
path = os.path.dirname(__file__)  # ? Directory path
assets_path = path + "/assets/samples/images/*"  # ? Assets path
output_folder_path = path + "/output"  # ? Output folder path

# * -- Functions


def clearConsole() -> None:  # ? Clear console
    os.system("cls" if os.name == "nt" else "clear")


def checkPaths() -> None:
    print(f"[{title}#checkPaths] path: ", path)
    print(f"[{title}#checkPaths] assets_path: ", assets_path)
    print(f"[{title}#checkPaths] output_folder_path: ", output_folder_path)


def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def lpr() -> None:
    print(f"[{title}#lpr] init")

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--psm", type=int, default=7,
        help="default PSM mode for OCR'ing license plates")
    args = vars(ap.parse_args())

    lprOperator = LPR(debug=False)

    files = glob.iglob(assets_path)
    print(f"[{title}#lpr] files: ", files)
    print("= = = = = = = = = = = = = = = = = = = = =")
    for file in files:
        try:
            print("- - - - - - - - - - - - - - - - - - - - -")
            print(f"[{title}#lpr] file: ", file)

            filename = file.split("/")[-1].split(".")[0]
            print(f"[{title}#lpr] filename: ", filename)

            image = cv2.imread(file)
            image = imutils.resize(image, width=600)
            (lpText, lpCnt) = lprOperator.find_and_ocr(image, psm=args["psm"],
                clearBorder=False)

            if lpText is not None:
                # output = "temp"
                output = cleanup_text(lpText)
                print(f"[{title}#lpr] ({filename}) output: ", output)

                output_destination = output_folder_path + f"/{filename}.txt"
                print(f"[{title}#lpr] output_destination: ", output_destination)

                outputFolderExists = os.path.exists(output_folder_path)
                print(f"[{title}#lpr] outputFolderExists: ", outputFolderExists)

                if not outputFolderExists:
                    print(f"[{title}#lpr] create output folder")
                    os.mkdir(output_folder_path)

                with open(output_destination, "w", encoding="utf-8") as result:
                    result.write(output)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)
            print(
                f"[{title}#lpr] ({filename}) error (line: {exc_tb.tb_lineno}): ", e)
    print("= = = = = = = = = = = = = = = = = = = = =")


#! Main
def main() -> None:
    clearConsole()

    print(f"[{title}#main] init")

    checkPaths()
    lpr()

    print(f"[{title}#main] exit")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
