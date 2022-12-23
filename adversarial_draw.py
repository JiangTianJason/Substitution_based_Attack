import matplotlib.pyplot as plt
from matplotlib import ticker
from pylab import *


x_axis_data = [1000, 2000, 3000, 4000, 5000]
wiki80_bert_axis_data = [0.6296,0.6511,0.6637,0.7054,0.7248]
wiki80_bertentity_axis_data = [0.5920,0.6351,0.6717,0.7126,0.7421]
wiki80_pcnn_axis_data = [0.5888,0.5888,0.6682,0.6869,0.6869]
tacred_bert_axis_data = [0.5872,0.5778,0.5952,0.6901,0.6953]
tacred_bertentity_axis_data = [0.5439,0.5624,0.6215,0.6663,0.6585]
tacred_pcnn_axis_data = [0.5699,0.6969,0.6930,0.7591,0.7876]



plt.plot(x_axis_data, wiki80_bert_axis_data, 'ro-', color='#1f77b4', alpha=0.8, linewidth=1, label='Wiki80_Bert')
plt.plot(x_axis_data, wiki80_bertentity_axis_data, 'bs-', color='#1f77b4', alpha=0.8, linewidth=1, label='Wiki80_BertEntity')
plt.plot(x_axis_data, wiki80_pcnn_axis_data, 'g^-', color='#1f77b4', alpha=0.8, linewidth=1, label='Wiki80_PCNN')
plt.plot(x_axis_data, tacred_bert_axis_data, 'ro-', color='#ff7f0e', alpha=0.8, linewidth=1, label='TACRED_Bert')
plt.plot(x_axis_data, tacred_bertentity_axis_data, 'bs-', color='#ff7f0e', alpha=0.8, linewidth=1, label='TACRED_BertEntity')
plt.plot(x_axis_data, tacred_pcnn_axis_data, 'g^-', color='#ff7f0e', alpha=0.8, linewidth=1, label='TACRED_PCNN')

plt.xticks(x_axis_data)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))


plt.legend(loc='center', bbox_to_anchor=(0.5,1.08),ncol=3,frameon = False)
plt.xlabel('Number of adversarial samples added to the training set')
plt.ylabel('Accuracy')

plt.grid(axis='y',linestyle = '--', linewidth = 0.5)
plt.savefig('adversarial_result_only_NV.jpg', bbox_inches='tight',dpi=600)

plt.show()
