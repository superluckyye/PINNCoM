from torchstat import stat
from DfpNet_FBS import TurbNetG, weights_init,TurbNetG_FBS,FBS_CNN, TurbNetG_student,TurbNetG_knowledge_distil
# model = TurbNetG(channelExponent=5)
model = TurbNetG(channelExponent=5)

# net_t = TurbNetG_knowledge_distil(channelExponent=expo, dropout=dropout)
stat(model,(3,128,128))