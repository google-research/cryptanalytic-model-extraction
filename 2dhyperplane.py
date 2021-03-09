from torch import nn
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

def plot(net):
	h = 0.005
	x_min, x_max = -1, 1
	y_min, y_max = -1, 1

	xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h),
							torch.arange(y_min, y_max, h))
	
	in_tensor = torch.cat((xx.reshape((-1,1)), yy.reshape((-1,1))), dim=1)

	z = net.forward(in_tensor)
	z = torch.argmax(z, dim=1)
	z = z.reshape(xx.shape)
	plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm)
	
	#weight = list(net.parameters())[0].data[0].numpy()
	#bias = list(net.parameters())[1].data[0].numpy()
	#x = np.linspace(-1,1,100)
	#y = [np.polyval(weight, [x, y]) for x, y in zip(xx, yy)]
	#plt.plot(x,y)
	plt.xlabel('x0')
	plt.ylabel('x1')
	
	plt.show()

class SimpleClassifier(nn.Module):
	def __init__(self, dims, activation):
		super().__init__()
		layers = OrderedDict()
		for i in range(len(dims) - 1):
			layers.update({f"fc{i}": nn.Linear(dims[i],dims[i+1])})
			layers.update({f"activation{i}": activation()})
		self.model = nn.Sequential(OrderedDict(layers))
		
	def forward(self,x):
		return self.model.forward(x)
		
model = SimpleClassifier([2, 3], nn.ReLU)
print(model)

plot(model)