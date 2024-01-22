import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18, remove_fc, model_urls
import torch.nn.functional as F

from transformers import RobertaModel

import math
from einops import rearrange
import torch.utils.model_zoo as model_zoo

from utils_new import RoBERTa_path


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class modality_specific_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(modality_specific_module, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        state_dict = remove_fc(model_zoo.load_url(model_urls[arch]))
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()
        assert arch == 'resnet50'
        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

        self.base.conv1 = None
        self.base.bn1 = None
        self.base.relu = None
        self.base.maxpool = None

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class DEE_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DEE_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x))/3
        x2 = self.FC2(F.relu(x2))
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)
        return out


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        
        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)
    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    ref: https://github.com/wudongming97/RMOT/blob/master/models/deformable_transformer_plus.py
    """

    def  __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        return x


class Visible2Text_module(nn.Module):
    def __init__(self, arch='resnet50', dataset='sysu'):
        super(Visible2Text_module, self).__init__()
        assert arch == 'resnet50'
        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        self.dataset = dataset
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.base.conv1 = None
        self.base.bn1 = None
        self.base.relu = None
        self.base.maxpool = None

        if dataset == 'regdb':
            self.base.layer4 = None

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        if self.dataset != 'regdb':
            x = self.base.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x


class visual_refinement_module(nn.Module):
    def __init__(self):
        super(visual_refinement_module, self).__init__()

    def forward(self, fv, ft, refine=True):
        """
        depthwise cross correlation
        ref: https://github.com/JudasDie/SOTS/blob/SOT/lib/models/sot/head.py#L227
        """
        if not refine:
            return fv

        # rearrange
        b, c, h, w = fv.size()
        x = rearrange(fv, 'b c h w -> b c (h w)')  # [B,C,HW]
        kernel = ft.unsqueeze(2)  # [B,C,1]

        # conv
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2))
        kernel = kernel.view(batch*channel, 1, kernel.size(2))
        out = F.conv1d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2))

        # rearrange
        out = rearrange(out, 'b c (h w) -> b c h w', h=h)  # [B,C,H,W]
        out += fv
        return out


class textual_refinement_module(nn.Module):
    def __init__(self):
        super(textual_refinement_module, self).__init__()

    def forward(self, fv, ft, refine=True):
        """
        channel attention
        """
        if not refine:
            return ft

        b, c, h, w = fv.size()
        fv = F.adaptive_avg_pool2d(fv, 1).view(b, c)  # [B,C]
        fv = F.sigmoid(fv)  # [B,C]
        out = ft + torch.mul(ft, fv)
        return out


class joint_embedding_module(nn.Module):
    def __init__(self, feat_dim):
        super(joint_embedding_module, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * feat_dim, 2 * feat_dim, kernel_size=1),
            nn.BatchNorm2d(2 * feat_dim),
        )

    def forward(self, fv, ft, relation=True):
        b, c, h, w = fv.size()
        if relation:
            ft = ft.view(b, c, 1, 1)  # [B,C,1,1]
            ft = ft.repeat([1, 1, h, w])  # [B,C,H,W]
            fvt = torch.cat([fv, ft], dim=1)  # [B,2C,H,W]
            fvt = fvt + self.conv(fvt)  # [B,2C,H,W]
            fvt = F.adaptive_avg_pool2d(fvt, 1).view(b, 2 * c)  # [B,2C]
        else:
            fv = F.adaptive_avg_pool2d(fv, 1).view(b, c)  # [B,C]
            fvt = torch.cat([fv, ft], dim=1)  # [B,2C]
        return fvt


class Joint_module(nn.Module):
    def __init__(self, feat_dim):
        super(Joint_module, self).__init__()
        # visual feature refinement
        self.visual_conv_1 = nn.Sequential(
            nn.Conv2d(3 * feat_dim, feat_dim, kernel_size=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.textual_fc_1 = nn.Linear(feat_dim, feat_dim)
        self.visual_refinement = visual_refinement_module()

        # textual feature refinement
        self.visual_conv_2 = nn.Sequential(
            nn.Conv2d(3 * feat_dim, feat_dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim//2, feat_dim, kernel_size=1),
        )
        self.textual_fc_2 = nn.Linear(feat_dim, feat_dim)
        self.textual_refinement = textual_refinement_module()

        # joint embedding
        self.joint_embedding = joint_embedding_module(feat_dim)


    def forward(self, fv, ft):
        """
        fv: visual feature, [B,3C,H,W]
        ft: textual feature, [B,C]
        """
        # visual feature refinement
        fv_1 = self.visual_conv_1(fv)  # [B,C,H,W]
        ft_1 = self.textual_fc_1(ft)  # [B,C]
        fv_refined = self.visual_refinement(fv_1, ft_1, refine=True)  # [B,C,H,W]

        # textual feature refinement
        fv_2 = self.visual_conv_2(fv)  # [B,C,H,W]
        ft_2 = self.textual_fc_2(ft)  # [B,C]
        ft_refined = self.textual_refinement(fv_2, ft_2, refine=True)  # [B,C]

        # joint embedding
        fvt = self.joint_embedding(fv_refined, ft_refined, relation=True)

        return fvt


class embed_net(nn.Module):
    def __init__(self, class_num, dataset, args):
        super(embed_net, self).__init__()

        self.thermal_module = modality_specific_module(arch=args.arch)
        self.visible_module = modality_specific_module(arch=args.arch)
        self.base_resnet = base_resnet(arch=args.arch)
        
        self.dataset = dataset
        if self.dataset == 'regdb': # For regdb dataset, we remove the MFA3 block and layer4.
            pool_dim = 1024
            self.DEE = DEE_module(512)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
            self.base_resnet.base.layer4 = None
        else:
            pool_dim = 2048
            self.DEE = DEE_module(1024)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
            self.MFA3 = MFA_block(1024, 512, 1)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.args = args
        self.text_mode = args.text_mode
        if self.text_mode in ('v1', 'v2'):
            self.text_encoder = RobertaModel.from_pretrained(RoBERTa_path, local_files_only=True)
            self.text_projection = FeatureResizer(
                input_feat_size=self.text_encoder.config.hidden_size,
                output_feat_size=pool_dim,
                dropout=True,
                do_ln=True,
            )
            self.visible_2_text = Visible2Text_module(arch=args.arch, dataset=dataset)
            self.visible_projection = FeatureResizer(
                input_feat_size=pool_dim,
                output_feat_size=pool_dim,
                dropout=True,
                do_ln=True,
            )

        if self.text_mode in ('v2',):
            self.joint_encoder = Joint_module(feat_dim=pool_dim)
            self.joint_bn = nn.BatchNorm1d(2 * pool_dim)
            self.joint_bn.bias.requires_grad_(False)  # no shift
            self.joint_bn.apply(weights_init_kaiming)
            self.joint_classifier = nn.Linear(2 * pool_dim, class_num, bias=False)
            self.joint_classifier.apply(weights_init_classifier)

    def forward_text(self, input_ids, attention_mask):
        encoded_text = self.text_encoder(input_ids, attention_mask)
        text_features = encoded_text.last_hidden_state
        text_features = self.text_projection(text_features)
        return text_features

    def forward(self, inputs, modal=0):
        x1, x2, input_ids, attention_mask = \
            inputs.get('visible_image'), inputs.get('thermal_image'), \
            inputs.get('input_ids'), inputs.get('attention_mask')

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x0 = torch.cat((x1, x2), 0)
            batch_size = x1.size(0)
        elif modal == 1:
            x0 = self.visible_module(x1)
            batch_size = x1.size(0)
        elif modal == 2:
            x0 = self.thermal_module(x2)
            batch_size = x2.size(0)

        x_ = x0
        x = self.base_resnet.base.layer1(x_)
        x_ = self.MFA1(x, x_)
        x = self.base_resnet.base.layer2(x_)
        x_ = self.MFA2(x, x_)
        if self.dataset == 'regdb':  # For regdb dataset, we remove the MFA3 block and layer4.
            xx = self.DEE(x_)
            x = self.base_resnet.base.layer3(xx)
        else:
            x = self.base_resnet.base.layer3(x_)
            xx = self.MFA3(x, x_)
            xx = self.DEE(xx)
            x = self.base_resnet.base.layer4(xx)
        
        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))
        
        feat = self.bottleneck(x_pool)  # [6B,C]

        txt_feat, v2t_feat, joint_feat = None, None, None

        if self.text_mode in ('v1', 'v2'):
            if modal in (0, 1):
                v2t_feat = self.visible_2_text(x0[:batch_size])  # [B,C]
                v2t_feat = self.visible_projection(v2t_feat)  # [B,C]
            if modal in (0, 2):
                word_feat = self.forward_text(input_ids, attention_mask)  # [B,L,C]
                txt_feat = torch.mean(word_feat, dim=1)  # [B,C]

        if self.text_mode in ('v2',):
            if self.training:
                f1, f2, f3, f4, f5, f6 = torch.chunk(x, 6, dim=0)
                rgb_feat = torch.cat([f1, f3, f5], dim=1)
                ir_feat = torch.cat([f2, f4, f6], dim=1)
            else:
                f1, f2, f3 = torch.chunk(x, 3, dim=0)
                rgb_feat = torch.cat([f1, f2, f3], dim=1)
                ir_feat = torch.cat([f1, f2, f3], dim=1)

            if modal in (0, 1):
                rgb_v2t_feat = self.joint_encoder(rgb_feat, v2t_feat)  # [B,2C]
            if modal in (0, 2):
                ir_txt_feat = self.joint_encoder(ir_feat, txt_feat)  # [B,2C]

            if modal == 0:
                joint_feat = torch.cat([rgb_v2t_feat, ir_txt_feat], dim=0)  # [2B,2C]
            elif modal == 1:
                joint_feat = rgb_v2t_feat  # [B,2C]
            elif modal == 2:
                joint_feat = ir_txt_feat  # [B,2C]

            joint_feat_bn = self.joint_bn(joint_feat)  # [B,2C]

        if self.training:
            return dict(
                feat=self.l2norm(x_pool) if self.args.use_amp else x_pool,
                logit=self.classifier(feat),
                txt_feat=self.l2norm(txt_feat) if (self.args.use_amp and txt_feat is not None) else txt_feat,
                v2t_feat=self.l2norm(v2t_feat) if (self.args.use_amp and v2t_feat is not None) else v2t_feat,
                joint_feat=self.l2norm(joint_feat) if (self.args.use_amp and joint_feat is not None) else joint_feat,
                joint_logit=self.joint_classifier(joint_feat_bn) if (self.args.use_amp and joint_feat is not None) else joint_feat,
            )
        else:
            return dict(
                feat_before_bn=self.l2norm(x_pool),
                feat_after_bn=self.l2norm(feat),
                txt_feat=self.l2norm(txt_feat) if txt_feat is not None else txt_feat,
                v2t_feat=self.l2norm(v2t_feat) if v2t_feat is not None else v2t_feat,
                joint_feat=self.l2norm(joint_feat) if joint_feat is not None else joint_feat,
            )