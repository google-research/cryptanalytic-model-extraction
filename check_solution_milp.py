# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sys
import pickle

np.random.seed(0)

NN = 0

QQ = 0
def n():
    global QQ
    QQ += 1
    return QQ

var = 0
allvars = []

def nvar(p='x'):
    global var
    res = p+str(var)
    allvars.append(res)
    var += 1
    return res

def dump_net(previous_outputs):
    global NN
    computation = []

    while len(allvars):
        allvars.pop()

    for i in range(len(A)):
        next_outputs = []
        for out_neuron in range(A[i].shape[1]):
            x_var = nvar()
            s_var = nvar('s')
            r_var = nvar('r')
            next_outputs.append(x_var)
            combination = " + ".join("%.17f * %s"%(a,b) for a,b in zip(A[i][:,out_neuron],previous_outputs))
            computation.append("s.t. ok_%d: "%n() + combination + " + " + str(B[i][out_neuron]) + " = " + x_var + " - " + s_var + ";")
            computation.append("s.t. relu_%d: %s <= 1000 * %s;" % (n(), x_var, r_var))
            computation.append("s.t. relu_%d: %s <= 1000 * (1 - %s);" % (n(), s_var, r_var))
    
        previous_outputs = next_outputs
    
    finalx = [x for x in allvars if x[0] == 'x'][-1]

    prefix = []
    
    for v in allvars:
        if v == finalx:
            prefix.append('var '+v+';')
        elif v[0] == 'x':
            prefix.append('var '+v+' >= 0;')
        elif v[0] == 's':
            prefix.append('var '+v+' >= 0;')
        elif v[0] == 'i':
            prefix.append('var '+v+' >= -1;')
        elif v[0] == 'r':
            prefix.append('var '+v+' binary;')
        else:
            raise

    NN += 1
    return prefix, computation[:-2] + ["s.t. final%d: %s = 0;"%(NN,[x for x in allvars if x[0] == 's'][-1])], finalx

    

name = sys.argv[1]

A, B = pickle.load(open("/tmp/real-%s.p"%name,"rb"))

sizes = [x.shape[0] for x in A] + [1]

inputs = [nvar('i') for _ in range(sizes[0])]

prefix1, rest1, outvar1 = dump_net(inputs)

A, B = pickle.load(open("/tmp/extracted-%s.p"%name,"rb"))

prefix2, rest2, outvar2 = dump_net(inputs)




import sys
sys.stdout=open("/tmp/test.mod","w")

print("\n".join('var '+v+' >= 0;' for v in inputs))

print("\n".join(prefix1))
print("\n\n")
print("\n".join(prefix2))

print("var slack;")
print("var which binary;")
print("maximize obj: %s-%s;"%(outvar1,outvar2))


print("\n".join(rest1))
print("\n".join(rest2))

for v in inputs:
    if v[0] == 'i':
        print("s.t. bounded%s: %s <= 1;"%(v,v))
    
print("solve;")
print('display %s;'%(", ".join(x for x in inputs)))
print('display %s;'%outvar1)
print('display %s;'%outvar2)
print('display slack;')

# Now it's on you. Go and run this model.
# you can export to mps with the following command.

# glpsol --check --wfreemps /tmp/o.mps --model /tmp/test.mod
