# Get git commit id
git_id=$(git rev-parse --short HEAD)
printf "$git_id\n"

# Create old_log directory
mkdir old_logs
dest_dir=./old_logs/logs-$git_id/
mkdir $dest_dir

# Compress log files.
cd logs
files=$(ls)
for filename in $files
do
    printf "zip and move $filename to old_logs...\n"
    zip -rq $filename.zip ./$filename
    mv -f $filename.zip ../$dest_dir
done
cd ..