# requires one command line argument that specifies the file name

# check for the correct number of arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi

idf.py monitor | tee "data/$1.out"