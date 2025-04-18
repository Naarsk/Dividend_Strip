import os                   # For file and directory operations
import subprocess           # To run smop as a subprocess
import shutil               # For moving files around

# Set the input root directory containing the MATLAB files
input_root = "APmodel"

# Set the output directory where converted Python files will go
output_root = "PyConverted_APmodel"

def convert_all_matlab_files(input_dir, output_dir):
    # Walk through all directories and files starting from input_dir
    for root, dirs, files in os.walk(input_dir):
        # Compute the relative path from the root input directory
        rel_path = os.path.relpath(root, input_dir)

        # Create the corresponding output path
        out_dir = os.path.join(output_dir, rel_path)

        # Make sure the output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # Loop over each file in the current folder
        for file in files:
            # Only process MATLAB files
            if file.endswith(".m"):
                # Full path to the input .m file
                input_file = os.path.join(root, file)

                # Get the base filename without the .m extension
                filename_no_ext = os.path.splitext(file)[0]

                # Expected Python output file name
                output_file = f"{filename_no_ext}.py"

                # Notify which file is being converted
                print(f"Converting: {input_file}")

                # If a .py file with the same name already exists in the current folder, delete it
                if os.path.exists(output_file):
                    os.remove(output_file)

                # Run smop on the .m file; no -o flag so output goes to current directory
                result = subprocess.run(["smop", input_file], capture_output=True, text=True)

                # If smop fails, print the error and continue to the next file
                if result.returncode != 0:
                    print(f"Error converting {input_file}:\n{result.stderr}")
                    continue

                # If smop produced the expected output .py file
                if os.path.exists(output_file):
                    # Move it to the correct location in the mirrored output directory
                    shutil.move(output_file, os.path.join(out_dir, output_file))
                else:
                    # If the output wasn't created, log a warning
                    print(f"SMOP did not produce output for: {input_file}")

# Run the function on the specified root directories
convert_all_matlab_files(input_root, output_root)
