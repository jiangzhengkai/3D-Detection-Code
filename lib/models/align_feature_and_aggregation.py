import torch
from torch import nn
from lib.csrc.alignfeature.modules.align_feature import AlignFeature
from lib.csrc.correlation.modules.correlation import Correlation

class Aggregation(nn.Module):
    def __init__(self,
                 num_channel,
                 name=''):
        super(Aggregation, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, align_feature, feature):

        feature_concat = torch.cat([align_feature, feature], dim=0)
        align_conv1_concat = self.conv1(feature_concat)
        align_conv1_concat = self.relu(align_conv1_concat)

        align_conv2_concat = self.conv2(align_conv1_concat)
        align_conv2_concat = self.relu(align_conv2_concat)

        align_conv3_concat = self.conv3(align_conv2_concat)
        align_conv3_concat = self.relu(align_conv3_concat)
        batch_size = align_feature.shape[0]

        align_conv3, feature_conv3 = torch.split(align_conv3_concat, batch_size, dim=0)

        weights = torch.cat([align_conv3, feature_conv3], dim=1)
        weights = torch.softmax(weights, dim=1)
        weights_slice = torch.split(weights, 1, dim=1)
        aggregation = weights_slice[0] * align_feature + weights_slice[1] * feature
        return aggregation


class Align_Feature_and_Aggregation(nn.Module):
    def __init__(self,
                 num_channel,
                 neighbor=9,
                 name=''):
        super(Align_Feature_and_Aggregation, self).__init__()
        self.num_channel = num_channel
        self.embed_conv1 = nn.Conv2d(num_channel, 64, 1)
        self.embed_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        import pdb;pdb.set_trace()
        self.align_feature = AlignFeature(neighbor, neighbor)
        self.relu = nn.ReLU()
        ######## correlation ########
        self.correlation = Correlation(kernel_size=1,
                                       patch_size=neighbor,
                                       stride=1,
                                       padding=0,
                                       dilation=1,
                                       dilation_patch=1)
        ######## aggregation ########
        self.aggregation = Aggregation(num_channel, name="Aggregation_Module")

    def forward(self, feature_select, feature_current):
        feature_concat = torch.cat([feature_select, feature_current], dim=0)
        import pdb;pdb.set_trace()
        embed_feature_concat_conv1 = self.embed_conv1(feature_concat)
        embed_feature_concat_relu1 = self.relu(embed_feature_concat_conv1)

        embed_feature_concat_conv2 = self.embed_conv2(embed_feature_concat_relu1)
        embed_feature_concat_relu2 = self.relu(embed_feature_concat_conv2)

        batch_size = feature_select.shape[0]
        embed_feature_current, embed_feature_select = torch.split(embed_feature_concat_relu2, batch_size, dim=0)

        weights = self.correlation(embed_feature_current, embed_feature_select)
        weights = weights.reshape([weights.shape[0],-1,weights.shape[3],weights.shape[4]])
        weights = torch.softmax(weights, dim=1)
        align_feature = self.align_feature(feature_select, weights)
        aggregation = self.aggregation(align_feature, feature_current)
        return aggregation
