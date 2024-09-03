import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Dual_Transformer
layer_group = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTM)

# Weight Initialize
def weights_init(m):
    if isinstance(m, layer_group):# isinstance() 函数检查模块 m 是否属于 layer_group 中的任何一种类型
        for name, param in m.named_parameters():#对模块 m 的每个参数进行迭代，named_parameters() 返回模块的所有参数及其名称。
            if 'weight' in name:#在参数名称中检查是否包含字符串 'weight'，以便只对权重进行初始化。
                nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('leaky_relu', 0.3))

class DilatedConv(nn.Module):
    def __init__(self, middle_channel,out_channel, twidth, fwidth,dilation):
        super(DilatedConv, self).__init__()
        self.middle_channel = middle_channel
        self.twidth = twidth
        self.fwidth = fwidth
        self.dilation = dilation

        self.pad_length1 = 1  * (fwidth - 1) 
        self.pad_length2 = dilation * (twidth - 1) 
        self.conv_layer = nn.Conv2d(middle_channel, out_channel, kernel_size=(twidth, fwidth), dilation=(dilation,1),bias=False)
        if twidth%2==1 and fwidth%2==1 :
            self.pad = nn.ConstantPad2d((self.pad_length1 // 2, self.pad_length1 // 2, self.pad_length2//2, self.pad_length2//2), 0)
        if twidth%2==0 and fwidth%2==0 :
            self.pad = nn.ConstantPad2d((self.pad_length1, 0, self.pad_length2, 0), 0)
    def forward(self, input_tensor):
        # Apply padding
        padded_input = self.pad(input_tensor)
        # Apply convolution
        output = self.conv_layer(padded_input)
        return output

class DeGateConv_audio(nn.Module):
    def __init__(self,input_channel: int,
                 middle_channel: int,
                 output_channel: int,
                 dilation=int
                 ):
        super(DeGateConv_audio, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        #self.pad1 = nn.ConstantPad2d((2, 2, 7, 7), value=0.)
       # self.pad2 = nn.ConstantPad2d((2, 2, 3, 3), value=0.)
        self.in_conv = nn.Sequential(nn.Conv2d(input_channel*2, middle_channel, kernel_size=(1, 1), stride=(1, 1),bias=False),
                                    nn.InstanceNorm2d(middle_channel, affine=True))
        self.in_conv11 =DilatedConv(middle_channel, middle_channel, twidth=3, fwidth=3,dilation=dilation)
        self.in_conv12 =DilatedConv(middle_channel, middle_channel, twidth=3, fwidth=3,dilation=dilation)

        self.in_conv21 =DilatedConv(middle_channel, middle_channel, twidth=2, fwidth=2,dilation=dilation)
        self.in_conv22 =DilatedConv(middle_channel, middle_channel, twidth=2, fwidth=2,dilation=dilation)

        self.sigmoid = nn.Sigmoid()
        self.SGA = SGA(middle_channel, epsilon=1e-5)
        self.out_conv =nn.Sequential(nn.ConvTranspose2d(middle_channel, output_channel, kernel_size=(2, 1), stride=(2, 1),bias=False),
                                    nn.InstanceNorm2d(output_channel, affine=True))
    def forward(self, x):
        x = F.leaky_relu(self.in_conv(x), negative_slope=0.3)
        # x11 = self.in_conv11(x) 
        # x12 = self.sigmoid(self.in_conv12(x11))
        
        # x21 = self.in_conv21(x)
        # x22 = self.sigmoid(self.in_conv22(x21))
        # x12_out = x11*x21
        # x22_out = x21*x12
        # x_out = x12_out+x22_out
        
        # out = self.SGA(x_out)
        out =  F.leaky_relu(self.out_conv(x), negative_slope=0.3)
        del x#, x11,x12,x21,x22,x12_out,x22_out
        return out

class enGateConv_audio(nn.Module):
    def __init__(self,input_channel: int,
                 middle_channel: int,
                 output_channel: int,
                 l_weigth:int,
                 ):
        super(enGateConv_audio, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        #self.pad1 = nn.ConstantPad2d((2, 2, 7, 7), value=0.)
       # self.pad2 = nn.ConstantPad2d((2, 2, 3, 3), value=0.)
        self.in_conv = nn.Sequential(nn.Conv2d(input_channel, middle_channel, kernel_size=(1, 1), stride=(1, 1),bias=False),
                                    nn.InstanceNorm2d(middle_channel, affine=True),nn.PReLU())
        self.in_conv11 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(l_weigth, 1), stride=(1, 1),padding =(l_weigth//2,0), bias=False)
        self.in_conv12 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(1, 5),  stride=(1, 1),padding =(0,2), bias=False)

        self.in_conv21 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(l_weigth//2+1, 3),  stride=(1, 1), padding =(l_weigth//4,1),bias=False)
        self.in_conv22 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(l_weigth//2+1, 3),  stride=(1, 1), padding =(l_weigth//4,1),bias=False)

        self.sigmoid = nn.Sigmoid()
        self.SGA = SGA(middle_channel, epsilon=1e-5)
        self.out_conv =nn.Sequential(nn.Conv2d(middle_channel*2, output_channel, kernel_size=(1, 1),stride=(2, 1), bias=False),
                                nn.InstanceNorm2d(output_channel, affine=True),nn.PReLU())
    def forward(self, x):
        x = self.in_conv(x)
        x11 = self.in_conv11(x) 
        x12 = self.sigmoid(self.in_conv12(x11))
        
        x21 = self.in_conv21(x)
        x22 = self.sigmoid(self.in_conv22(x21))
        x12_out = x11*x21
        x22_out = x21*x12
        x_out = torch.cat((x12_out,x22_out),dim=1)
        
        #out = self.SGA(x_out)
        out = self.out_conv(x_out)
        del x, x11,x12,x21,x22,x12_out,x22_out
        return out

class SGA(nn.Module):
    def __init__(self, channel, epsilon=1e-5):
        super(SGA, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.epsilon = epsilon

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate =torch.tanh(embedding * norm + self.beta)
        return x + gate

class GateConv(nn.Module):
    def __init__(self,input_channel: int,
                 middle_channel: int,
                 output_channel: int,
                 ):
        super(GateConv, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.a = nn.Parameter(torch.Tensor([0]))
        self.b = nn.Parameter(torch.Tensor([0]))
        self.in_conv11 =nn.Sequential(nn.Conv1d(input_channel, middle_channel, kernel_size=(13), padding=(6),stride=2,bias=False),
                                        nn.InstanceNorm1d(middle_channel, affine=True),nn.PReLU())
        self.in_conv12 =nn.Sequential(nn.Conv1d(middle_channel, output_channel, kernel_size=(13), padding=(6),stride=2,bias=False),
                                        nn.InstanceNorm1d(output_channel, affine=True),nn.PReLU())

        self.in_conv31 =nn.Conv1d(output_channel, output_channel, kernel_size=(9), padding=(4),stride=2,bias=False)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.in_conv32 =nn.Conv1d(output_channel, output_channel, kernel_size=(9), padding=(4),stride=2,bias=False)

        self.in_conv2 =nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size=(9), padding=(4),stride=2,bias=False),
                                        nn.InstanceNorm1d(output_channel, affine=True),nn.PReLU())

        self.out_conv =nn.Sequential(nn.Conv1d(middle_channel, output_channel, kernel_size=(1), bias=False),
                                    nn.InstanceNorm1d(output_channel, affine=True),nn.PReLU())
    def forward(self,x1,x2,x3):
        x1 = self.in_conv12(self.in_conv11(x1))
        x2 =self.in_conv2(x2)
        x_out = self.in_conv31(self.avg_pool(self.in_conv31(x1 + x2)))
        x_out = torch.sigmoid(x_out)*(x1 + x2)
        out = self.a*x_out + self.b*x3
        del x1, x2, x3,x_out
        return out

class SAVSE(torch.nn.Module):  #propose+ 融合的通道注意力
    def __init__(self, frame_seq):
        super(SAVSE, self).__init__()
        self.frame_seq = frame_seq

        self.a_conv1 =enGateConv_audio(input_channel=1,middle_channel=32, output_channel=32,l_weigth=13)
        #self.a_pool1 = nn.MaxPool2d((2, 1))
        self.a_conv2 =enGateConv_audio(input_channel=32,middle_channel=32, output_channel=32,l_weigth=9)
        self.a_conv3 =enGateConv_audio(input_channel=32,middle_channel=32, output_channel=32,l_weigth=5)

        self.AV_attention = AV_MiltiHeadAttention(dim=32, num_heads=4, qkv_bias = False, qk_scale= None, attn_drop= 0.2, proj_drop= 0.2)
        self.image_process = Image_process(in_channel=1,middle_channel=32,out_channel=32)

        self.gateConv = GateConv(input_channel=32,middle_channel=64,output_channel=32)
        self.av_lstm2 = nn.LSTM(input_size=1024, hidden_size=128, num_layers=1, bias=False, batch_first=True, dropout=0, bidirectional=True)
        self.av_fc2 = nn.Linear(256 * frame_seq, 512, bias=False)
        self.av_fc2_ln = nn.LayerNorm(512, elementwise_affine=True)

        self.a_fc3 = nn.Linear(512, 1 * 256, bias=False)
        self.v_fc3 = nn.Linear(512, 2048, bias=False)

    def forward(self, noisy,lip):#
        B,_,_,_= noisy.shape 
        noisy1 = self.a_conv1(noisy)#audio-(1,32,257,5)
        noisy2= self.a_conv2(noisy1)#audio-(1,16,128,5)
        noisy3 = self.a_conv3(noisy2)

        noisy1 = noisy1.permute(0 ,3, 2, 1) # -> (batch_size*5, frame_seq, c)
        noisy1 = torch.flatten(noisy1, start_dim=0, end_dim=1)
        noisy2 = noisy2.permute(0 ,3, 2, 1) # -> (batch_size*5, frame_seq, c)
        noisy2 = torch.flatten(noisy2, start_dim=0, end_dim=1)
        noisy3 = noisy3.permute(0 ,3, 2, 1) # -> ((batch_size*5, frame_seq, c)
        noisy3 = torch.flatten(noisy3, start_dim=0, end_dim=1)

        lip1,lip2,lip3 = self.image_process(lip)     # (batch_size*5, frame_seq, c)

        #av_list = []
        av1 = self.AV_attention(noisy1,lip1)
        #av_list.append(av1)
        av2 = self.AV_attention(noisy2,lip2)#(batch_size*5,F, C)
        #av_list.append(av2)
        av3 = self.AV_attention(noisy3,lip3)
        #av_list.append(av3)

        #av3 = av3.view(B,5,-1,32)
        av3 = av3.permute(0, 2, 1)
        #av2 = av2.view(B,5,-1,32)
        av2 = av2.permute(0, 2, 1)
        #av1= av1.view(B,5,-1,32)
        av1 = av1.permute(0, 2, 1)

        av= self.gateConv(av1,av2,av3)
        av = torch.flatten(av, start_dim=1, end_dim=2)
        av = av.view(B,5,-1)
        self.av_lstm2.flatten_parameters()

        x, (hn, cn) = self.av_lstm2(av) # -> (batch_size, frame_seq, 512)
        x = torch.flatten(x, start_dim=1) # -> (batch_size, 2048)
        x = self.av_fc2(x) # -> (batch_size, 512)
        x = self.av_fc2_ln(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        noisy = self.a_fc3(x)
        noisy = F.leaky_relu(noisy, negative_slope=0.3, inplace=True)
        noisy = noisy.view(-1, 1, 256)

        #lip = self.v_fc3(x)
        #lip = F.leaky_relu(lip, negative_slope=0.3, inplace=True)
        #lip = lip.view(-1, 2048)

        return noisy#, lip

class Image_process(torch.nn.Module):
    # input: lip feature matrix extracted with pretrained autoencoder
    # (batch_size,C, H, W, frame_seq)
    # (batch_size, 2048, frame_seq)

    def __init__(self, in_channel, middle_channel, out_channel):
        super(Image_process, self).__init__()
        self.Conv11 = nn.Conv2d(in_channel, middle_channel, kernel_size=(5, 5), padding=(2, 2),stride=(2,1),bias=False)# ,stride=(2,2)
        self.Conv12 = nn.Conv2d(middle_channel, middle_channel, kernel_size=(5, 5), padding=(2, 2),stride=(1,2),bias=False)# ,stride=(2,2)
        self.Conv2 = nn.Sequential(nn.Conv2d(middle_channel, out_channel, kernel_size=(5, 5),padding=(2, 2), stride=(2,1), bias=False),
                                      nn.InstanceNorm2d(out_channel, affine=True),nn.PReLU())
        self.Conv3 = nn.Conv2d(out_channel, middle_channel, kernel_size=(5, 5), padding=(2, 2),stride=(1,2),bias=False)
        self.Conv4 = nn.Sequential(nn.Conv2d(middle_channel, out_channel, kernel_size=(5, 5), padding=(2, 2),bias=False),
                                nn.InstanceNorm2d(out_channel, affine=True),nn.PReLU())
        self.Conv5 = nn.Conv2d(out_channel, middle_channel, kernel_size=(3, 3),padding=(1, 1),stride=(2,1),bias=False)#,stride=(2,2)
        self.Conv6 = nn.Sequential(nn.Conv2d(middle_channel, out_channel, kernel_size=(3, 3),padding=(1, 1),bias=False),
                                    nn.InstanceNorm2d(out_channel, affine=True),nn.PReLU())
    def forward(self, lip):#lip-(frame_num, 16, 16, 3)
        B, _, _,_,_= lip.shape 
        lip = lip.permute(-1,0,1,2,3)
        lip = torch.flatten(lip, start_dim=0, end_dim=1)
        lip1 = self.Conv2(self.Conv12(self.Conv11(lip)))
        lip2 = self.Conv4(self.Conv3(lip1))
        lip3 = self.Conv6(self.Conv5(lip2))##audio-(B*5,C,H,W)

        lip1 = torch.flatten(lip1, start_dim=2, end_dim=3)#audio-(B*5,C,H*W)
        lip1 = lip1.permute(0, 2, 1) # -> (batch_size, frame_seq, 2048)-(1,5,2048)    (b*5,f,c)  

        lip2 = torch.flatten(lip2, start_dim=2, end_dim=3)
        lip2 = lip2.permute(0, 2, 1)

        lip3 = torch.flatten(lip3, start_dim=2, end_dim=3)
        lip3 = lip3.permute(0, 2, 1)

        return lip1,lip2,lip3

class SELayer(nn.Module):  # [b, 32, t, 256]  # 第二篇放入MSA里面 # [f, b*t, c]
    def __init__(self, channel):
        super(SELayer, self).__init__()

        self.liear1 = nn.Linear(channel, channel, bias=False)
        self.liear2 = nn.Linear(channel, channel, bias=False)

        #self.avg_pool = torch.mean(x_out1, dim=-1)
        #self.max_pool = torch.max(x_out1, dim=-1)
        self.avg_pool = nn.AdaptiveAvgPool1d(channel)
        self.max_pool = nn.AdaptiveMaxPool1d(channel)
        self.temperature = channel ** 0.5  # 根号d  ==d_k(dim_head) ** 0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, k):
        q = self.liear1(q)
        k = self.liear1(k)
        x = torch.bmm(q.transpose(1, 2)/self.temperature, k)  # # [f, B*T, B*T]
        y1 = self.avg_pool(x)   
        y2 = self.max_pool(x)   
        #y1 = torch.mean(x, dim=-1).unsqueeze(-1)
        #y2 = torch.max(x, dim=-1).unsqueeze(-1)
        y = y1 + y2
        y =  self.sigmoid(y)
        return y

class AV_MiltiHeadAttention(nn.Module):
    def __init__(self,dim,num_heads=4,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super(AV_MiltiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #这里的dim指的是c，即input输入N个数据的维度。
        # 另外，为避免繁琐，将8个头的权重拼在一起，而三个不同的权重由后面的Linear生成。
        # 而self.qkv的作用是是将input X (N,C)与权重W（C，8*Ｃ1*3）相乘得到Q_K_V拼接在一起的矩阵。
        # 所以,dim*3表示的就是所有头所有权重拼接的维度，即8*Ｃ1*3。即dim=C=C1*3。
        self.scale = qk_scale or head_dim ** -0.5
        self.lip_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = nn.Linear(dim,dim *3,bias = qkv_bias)
        #bias默认为True，为了使该层学习额外的偏置。
        self.attn_drop = nn.Dropout(attn_drop)
        #dropout忽略一半节点
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.eca_attn = SELayer(dim)
    def forward(self, audia, lip):
        B,N,C = audia.shape
        #B为batch_size
        qkv = self.qkv(audia).reshape(B,N,3,self.num_heads,C // self.num_heads).permute(2,0,3,1,4)
        lip = self.lip_v(lip).reshape(B, N, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        #将x构造成一个（B，N，3，8，c1）的一个矩阵，然后将其变成（3，B，8，N，c1）。
        #是为了使得后面能将其分为三部分，分别作为不同的权重，维度为（B，8，N，c1）
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = ((q @ k.transpose(-2,-1)) * self.scale)
        #将k的维度从（B，8，N，c1）转变为（B，8，c1，N），其实就是对单头key矩阵转置，使Query*key^T，得到scores结果，然后×self.scale，即×C1的-0.5。
        #就是做一个归一化。乘一个参数，不一定是这个值。
        #维度为（B，8，N，N）
        attn_qk = attn.softmax(dim = -1) @ k
        attn_av = (attn_qk * lip).softmax(dim = -1)
        #对归一化的scores做softmax
        # 维度为（B，8，N，N）
        av_out = (attn_qk * attn_av).transpose(1,2).reshape(B,N,C)
        #将scores与values矩阵相乘，得到数据维度为（B，8，N，C1），使用transpose将其维度转换为（B，N，8，C1）
        av_out = self.proj(av_out)
        #做一个全连接
        av_out=self.proj_drop(av_out)

        av_attn = self.eca_attn(audia,audia)
        av_out=torch.bmm(av_out,av_attn)
        #做Dropout
        return av_out
    

class LAVSE(torch.nn.Module):
    # input: spectrogram, lip feature matrix extracted with pretrained autoencoder
    # input size: (batch_size, 1, 257, frame_seq), (batch_size, 2048, frame_seq)
    # output size: (batch_size, 1, 257), (batch_size, 2048)

    def __init__(self, frame_seq):
        super(LAVSE, self).__init__()
        self.frame_seq = frame_seq

        self.a_conv1 = nn.Conv2d(1, 32, (25, 5), 1, (12, 2), bias=False)
        self.a_conv1_in = nn.InstanceNorm2d(32, affine=True)

        self.a_pool2 = nn.MaxPool2d((2, 1))

        self.a_conv3 = nn.Conv2d(32, 32, (17, 5), 1, (8, 2), bias=False)
        self.a_conv3_in = nn.InstanceNorm2d(32, affine=True)

        self.a_conv4 = nn.Conv2d(32, 16, (9, 5), 1, (4, 2), bias=False)
        self.a_conv4_in = nn.InstanceNorm2d(16, affine=True)
        self.image_process = Image_process(in_channel=1,middle_channel=32,out_channel=32)

        self.av_lstm2 = nn.LSTM(input_size=4096, hidden_size=256, num_layers=1, bias=False, batch_first=True, dropout=0, bidirectional=True)
        self.av_fc2 = nn.Linear(512 * frame_seq, 512, bias=False)
        self.av_fc2_ln = nn.LayerNorm(512, elementwise_affine=True)

        self.a_fc3 = nn.Linear(512, 1 * 257, bias=False)
        #self.v_fc3 = nn.Linear(512, 2048, bias=False)

    def forward(self, noisy,lip):
        noisy = self.a_conv1(noisy)
        noisy = self.a_conv1_in(noisy)
        noisy = F.leaky_relu(noisy, negative_slope=0.3)

        noisy = self.a_pool2(noisy)

        noisy = self.a_conv3(noisy)
        noisy = self.a_conv3_in(noisy)
        noisy = F.leaky_relu(noisy, negative_slope=0.3)

        noisy = self.a_conv4(noisy)
        noisy = self.a_conv4_in(noisy)
        noisy = F.leaky_relu(noisy, negative_slope=0.3)

        noisy = torch.flatten(noisy, start_dim=1, end_dim=2)
        noisy = noisy.permute(0, 2, 1) # -> (batch_size, frame_seq, 2048)

        lip = self.image_process(lip)
       # lip = lip.permute(0, 2, 1) # -> (batch_size, frame_seq, 2048)

        x = torch.cat((noisy, lip), 2) # -> (batch_size, frame_seq, 4096)

        self.av_lstm2.flatten_parameters()
        x, (hn, cn) = self.av_lstm2(x) # -> (batch_size, frame_seq, 512)
        x = torch.flatten(x, start_dim=1) # -> (batch_size, 2560)
        x = self.av_fc2(x) # -> (batch_size, 512)
        x = self.av_fc2_ln(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        noisy = self.a_fc3(x)
        noisy = F.leaky_relu(noisy, negative_slope=0.3, inplace=True)
        noisy = noisy.view(-1, 1, 257)

        #lip = self.v_fc3(x)
        #lip = F.leaky_relu(lip, negative_slope=0.3, inplace=True)
        #lip = lip.view(-1, 2048)

        return noisy#, lip
    