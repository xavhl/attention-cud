program_path="."
configs=(
    1 8 8
    8 32 32
    16 64 64
    32 128 128
    64 256 256
    128 512 512
)
num_configs=${#configs[@]}
output_path="./output/output_tests.txt"
> $output_path # clear content

# test 1
printf "*** test 1: sequential vs openmp vs cuda ***\n\n"
printf "*** test 1: sequential vs openmp vs cuda ***\n\n" >> $output_path
programs_t1=("attention_sequential" "attention_openmp"  "attention_cuda")

for program_file in ${programs_t1[@]}; do
    printf "%s\n" $program_file; printf "%s\n" $program_file >> $output_path
    for ((i=0; i<num_configs; i+=3)); do
        b="${configs[i]}"
        s="${configs[i+1]}"
        e="${configs[i+2]}"
        output=$($program_path/$program_file -b $b -s $s -e $e)
        printf "config=(%s,%s,%s): %s\n" $b $s $e $output; printf "config=(%s,%s,%s): %s\n" $b $s $e $output >> $output_path
    done
    printf "\n"; printf "\n" >> $output_path
done

# test 2
printf "*** test 2: simple vs transposed vs tiling (thread) vs tiling (block) vs flash vs flash multi ***\n\n"
printf "*** test 2: simple vs transposed vs tiling (thread) vs tiling (block) vs flash vs flash multi ***\n\n" >> $output_path
programs_t2=("attention_cuda -m t" "attention_cuda_tile" "attention_cuda_tile -m r" "attention_flash" "attention_flash -m m")

for p in {0..4}; do
    program_file=${programs_t2[p]}
    printf "%s\n" $program_file; printf "%s\n" $program_file >> $output_path
    for ((i=0; i<num_configs; i+=3)); do
        b="${configs[i]}"
        s="${configs[i+1]}"
        e="${configs[i+2]}"
        output=$($program_path/$program_file -b $b -s $s -e $e)
        printf "config=(%s,%s,%s): %s\n" $b $s $e $output; printf "config=(%s,%s,%s): %s\n" $b $s $e $output >> $output_path
    done
    printf "\n"; printf "\n" >> $output_path
done

# test 3
printf "*** test 3 (unified memory): tiling (block) vs flash vs flash multi ***\n\n"
printf "*** test 3 (unified memory): tiling (block) vs flash vs flash multi ***\n\n" >> $output_path
programs_t3=("attention_cuda_tile -m r -u" "attention_flash -u" "attention_flash -m m -u")

for p in {0..2}; do
    program_file=${programs_t3[p]}
    printf "%s\n" $program_file; printf "%s\n" $program_file >> $output_path
    for ((i=0; i<num_configs; i+=3)); do
        b="${configs[i]}"
        s="${configs[i+1]}"
        e="${configs[i+2]}"
        output=$($program_path/$program_file -b $b -s $s -e $e)
        printf "config=(%s,%s,%s): %s\n" $b $s $e $output; printf "config=(%s,%s,%s): %s\n" $b $s $e $output >> $output_path
    done
    printf "\n"; printf "\n" >> $output_path
done

# test 4
printf "*** test 4 (tile size): tiling (thread) vs tiling (block) ***\n\n"
printf "*** test 4 (tile size): tiling (thread) vs tiling (block) ***\n\n" >> $output_path
programs_t4=("attention_cuda_tile" "attention_cuda_tile -m r")
tile_sizes=(1 4 8 16 32) # tile_sizes=(4 8 16)
b=128; s=512; e=512 # b=1; s=2; e=2
printf "b=%s s=%s e=%s\n\n" $b $s $e; printf "b=%s s=%s e=%s\n\n" $b $s $e >> $output_path

for p in {0..1}; do
    program_file=${programs_t4[p]}
    printf "%s\n" $program_file; printf "%s\n" $program_file >> $output_path
    for tile_size in ${tile_sizes[@]}; do
        output=$($program_path/$program_file -b $b -s $s -e $e -w $tile_size)
        printf "tile=%s: %s\n" $tile_size $output; printf "tile=%s: %s\n" $tile_size $output >> $output_path
    done
    printf "\n"; printf "\n" >> $output_path
done

# test 5
printf "*** test 5 (separate operations) using \"tiling (block)\" ***\n\n"
printf "*** test 5 (separate operations) using \"tiling (block)\" ***\n\n" >> $output_path
program_file="attention_cuda_tile -m r"
printf "attention_cuda_tile -m r\n\n" $program_file; printf "attention_cuda_tile -m r\n\n" $program_file >> $output_path

for ((i=0; i<num_configs; i+=3)); do
    b="${configs[i]}"
    s="${configs[i+1]}"
    e="${configs[i+2]}"
    output=$($program_path/$program_file -b $b -s $s -e $e -p)
    printf "config=(%s,%s,%s): %s %s %s %s\n" $b $s $e $output; printf "config=(%s,%s,%s): %s %s %s %s\n" $b $s $e $output >> $output_path
done