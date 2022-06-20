import unpack_barspec_fh as ub
from pathlib import Path
import sys

momdict = {
    "op2_q+6+4+2_op2_q-6-4-2": [[-3, -2, -1], [3, 2, 1]],
    "op8_q+6+4+2_op8_q-6-4-2": [[-3, -2, -1], [3, 2, 1]],
    "op2_q+6+0+0_op2_q-6+0+0": [[-3, 0, 0], [3, 0, 0]],
    "op8_q+6+0+0_op8_q-6+0+0": [[-3, 0, 0], [3, 0, 0]],
    "op2_q+4+4+2_op2_q-4-4-2": [[-2, -2, -1], [2, 2, 1]],
    "op8_q+4+4+2_op8_q-4-4-2": [[-2, -2, -1], [2, 2, 1]],
    "op2_q+4+2+2_op2_q-4-2-2": [[-2, -1, -1], [2, 1, 1]],
    "op8_q+4+2+2_op8_q-4-2-2": [[-2, -1, -1], [2, 1, 1]],
    "op2_q+2+2+0_op2_q-2-2+0": [[-1, -1, 0], [1, 1, 0]],
    "op8_q+2+2+0_op8_q-2-2+0": [[-1, -1, 0], [1, 1, 0]],
    "op8_q+0+0+0_op8_q+0+0+0": [[0, 0, 0]],
    "": [
        [-3, -2, -1],
        [-3, 0, 0],
        [-2, -2, -1],
        [-2, -1, -1],
        [-1, -1, 0],
        [0, 0, 0],
        [1, 1, 0],
        [2, 1, 1],
        [2, 2, 1],
        [3, 0, 0],
        [3, 2, 1],
    ],
}

# Give a file containing a list of limefiles as input
if len(sys.argv)>1:
    filelist_name = sys.argv[1]
else: 
    filelist_name = 'filelist_b5p65kp122130kp121756.lst' # Default filelist 

with open(filelist_name,'r') as f:
    filelist_comb = [line.strip() for line in f]
        
base_dir = Path("/scratch/usr/hhpmbate/limepickle/")
data_dir = base_dir / Path(filelist_name[9:-4])
data_dir.mkdir(parents=True, exist_ok=True)
print("\nSaving data in: ", data_dir)

ub.unpack_barspec_FH(
    filelist_comb,
    loc=data_dir.as_posix(),
    momdict=momdict,
)
