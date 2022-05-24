for index in 328 159 134 174
do
  for filename in $(seq $index 720 14400)
  do
    printf "copy $filename\n"
    cp -r ./logs/mc_medqn2/$filename/ ./logs-mc/mc_medqn2/
  done
done