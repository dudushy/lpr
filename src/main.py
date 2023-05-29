# pip freeze | xargs pip uninstall -y
# pip freeze > requirements.txt
# * -- Imports
import os
import glob


# * -- Variables
title = "LPR"
path = os.path.dirname(__file__)  # ? Directory path
assets_path = path + r"\assets\samples\*"  # ? Assets path
output_folder = path + r"\assets\output"  # ? Assets output folder path

# * -- Functions


def clearConsole() -> None:  # ? Clear console
    os.system("cls" if os.name == "nt" else "clear")


def test() -> None:
    print(f"[{title}#test] path: ", path)
    print(f"[{title}#test] assets_path: ", assets_path)
    print(f"[{title}#test] output_folder: ", output_folder)


def lpr() -> None:
    files = glob.iglob(assets_path)
    print(f"[{title}#lpr] files: ", files)
    print("= = = = = = = = = = = = = = = = = = = = =")
    for file in files:
        try:
            print("- - - - - - - - - - - - - - - - - - - - -")
            print(f"[{title}#lpr] file: ", file)

            filename = file.split("\\")[-1].split(".")[0]
            print(f"[{title}#lpr] filename: ", filename)

            # TODO: LPR

            output = "temp"
            print(f"[{title}#lpr] ({filename}) output: ", output)

            output_destination = output_folder + f"\\{filename}.txt"
            print(f"[{title}#lpr] output_destination: ", output_destination)

            outputFolderExists = os.path.exists(output_folder)
            print(f"[{title}#lpr] outputFolderExists: ", outputFolderExists)

            if not outputFolderExists:
                print(f"[{title}#lpr] create output folder")
                os.mkdir(output_folder)

            with open(output_destination, "w", encoding="utf-8") as result:
                result.write(output)
        except Exception as e:
            # print("\n" + e)
            print(f"[{title}#lpr] ({filename}) error: ", e)
    print("= = = = = = = = = = = = = = = = = = = = =")


def main() -> None:
    clearConsole()
    test()
    lpr()


#! Main
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
