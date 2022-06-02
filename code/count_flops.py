import subprocess
from statistics import median


def measure_flops(args):
    args = args.split()
    perf_str =  "perf stat -e cycles -e mem-loads -e mem-stores -e fp_arith_inst_retired.scalar_single -e fp_arith_inst_retired.scalar_double -e fp_arith_inst_retired.256b_packed_single -e fp_arith_inst_retired.scalar_single -e fp_arith_inst_retired.256b_packed_single -e fp_arith_inst_retired.256b_packed_double -e fp_arith_inst_retired.128b_packed_single -e fp_arith_inst_retired.128b_packed_double -e dTLB-load-misses -e dTLB-load -e dTLB-store-misses -e dTLB-stores -e L1-dcache-load-misses -e L1-dcache-loads"
    perf_cmd = perf_str.split(" ") + ["./shapley_values"] + args
    proc = subprocess.run(perf_cmd, capture_output=True)

    perf_output = proc.stderr.decode("utf-8")
    flops = 0
    for line in perf_output.split("\n"):
        line = line.strip()
        #print(line)

        try:
            nums = int(line.split("      ")[0].replace(",", ""))
            name = line.split("      ")[1]

            if name == "fp_arith_inst_retired.scalar_single:u" or name == "fp_arith_inst_retired.scalar_double:u":
                flops += nums
            elif name == "fp_arith_inst_retired.128b_packed_single:u" or name == "fp_arith_inst_retired.128b_packed_double:u":
                flops += 2 * nums
            elif name == "fp_arith_inst_retired.256b_packed_single:u" or name == "fp_arith_inst_retired.256b_packed_double:u":
                flops += 4 * nums
            
        except:
            pass
    
    return flops



print("flops_per_input_size = {")
input_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
for i in input_sizes:
    measured_flops = []
    for run in range(3):
        measured_flops.append(measure_flops(f"-i {i} -n 1"))
    
    m = median(measured_flops)
    print(f"    {i}: {m},")

print("}")


#print("Arguments: ./shapley_values ", end="")
#print("Flops: ", measure_flops(input()))



#print("FLOPS: ", flops)
