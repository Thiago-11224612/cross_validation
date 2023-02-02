import matplotlib.pyplot as plt
import numpy as np
import os


NRMS, SDQ, SDPr, SASr, side_labels = [], [], [], [], []
with open('NRMS.txt', 'r') as file:
	num = []
	lines = file.readlines()
	for line in lines:
		line = line.split(' ')
		num_lin = []
		for elem in line:
			try:
				num_lin.append(float(elem))
			except ValueError:
				print()
		num.append(num_lin)
print(num)
for i in num:
	side_labels.append(str(int(i[0])))
	NRMS.append(i[1])
	SDQ.append(i[2])
	SDPr.append(i[3])
	SASr.append(i[1]+i[4])

axis = np.arange(len(side_labels))
fig, ax = plt.subplots()

ax.set_xticks(axis,side_labels)


ax.set_xlabel('Número de pixels por lado')
ax.set_ylabel(r'Campo elétrico do sinal ($watt^{\frac{1}{2}}$)')
ax.text(0.5,20,'azul: SDQ\nvermelho: SDPr+SDQ = NRMS\nverde: SASr+SDPr+SDQ (erro da dist. uniforme)', fontsize = 'medium')
plt.margins(x=0)
plt.plot(axis, SDQ, 'bo')
plt.plot(axis, NRMS, 'ro')
plt.plot(axis, SASr, 'go')
plt.savefig('NRMS plot.png')