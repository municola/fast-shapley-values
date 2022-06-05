import subprocess
import sys
import tqdm
from statistics import median


if len(sys.argv) == 1:
    shapley_cmd = "./shapley_values"
    impl = "exact"
else:
    shapley_cmd = sys.argv[1]
    impl = sys.argv[2]


print("shapley_cmd", shapley_cmd)
print("impl", impl)

def measure_flops(args):
    args = args.split()
    perf_str =  "perf stat -e cycles -e mem-loads -e mem-stores -e fp_arith_inst_retired.scalar_single -e fp_arith_inst_retired.scalar_double -e fp_arith_inst_retired.256b_packed_single -e fp_arith_inst_retired.scalar_single -e fp_arith_inst_retired.256b_packed_single -e fp_arith_inst_retired.256b_packed_double -e fp_arith_inst_retired.128b_packed_single -e fp_arith_inst_retired.128b_packed_double -e dTLB-load-misses -e dTLB-load -e dTLB-store-misses -e dTLB-stores -e L1-dcache-load-misses -e L1-dcache-loads"
    perf_cmd = perf_str.split(" ") + [shapley_cmd] + args
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



out = "flops_per_input_size = {\n"
input_sizes = [256*i for i in range(1, 8192//256+1)]
#input_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

for i in tqdm.tqdm(input_sizes):
    measured_flops = []
    for run in range(2):
        measured_flops.append(measure_flops(f"-i {i} -n 1 --impl {impl}"))
    
    m = median(measured_flops)
    out += f"    {i}: {m},\n"

out += "}\n"

print(out)

#print("Arguments: ./shapley_values ", end="")
#print("Flops: ", measure_flops(input()))



#print("FLOPS: ", flops)
