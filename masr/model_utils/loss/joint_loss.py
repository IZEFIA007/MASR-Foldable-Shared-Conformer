import torch
import torch.nn as nn


class JointTrainingLoss(nn.Module):
    """可折叠Conformer的联合训练损失函数"""

    def __init__(self, alpha_p=0.25, ctc_weight=0.3, lsm_weight=0.1):
        super().__init__()
        self.alpha_p = alpha_p
        self.ctc_weight = ctc_weight
        self.lsm_weight = lsm_weight

        # 基础损失函数
        self.ctc_loss = CTCLoss()
        self.att_loss = LabelSmoothingLoss(
            smoothing=lsm_weight,
            normalize_length=False
        )

    def forward(self, encoder_out_max, encoder_out_seed, targets, target_lengths,
                encoder_out_lens_max, encoder_out_lens_seed, reverse_weight=0.0):
        """
        计算联合训练损失

        Args:
            encoder_out_max: 最大展开模型的编码器输出
            encoder_out_seed: 种子模型的编码器输出
            targets: 目标文本
            target_lengths: 目标文本长度
            encoder_out_lens_max: 最大展开模型的输出长度
            encoder_out_lens_seed: 种子模型的输出长度
        """
        # 1. 计算最大展开模型的CTC损失
        ctc_loss_max = self.ctc_loss(
            encoder_out_max, encoder_out_lens_max, targets, target_lengths
        )

        # 2. 计算种子模型的CTC损失
        ctc_loss_seed = self.ctc_loss(
            encoder_out_seed, encoder_out_lens_seed, targets, target_lengths
        )

        # 3. 计算联合CTC损失
        joint_ctc_loss = ctc_loss_max + self.alpha_p * ctc_loss_seed

        # 4. 如果需要，还可以添加Attention损失的联合计算
        # att_loss_max = ...
        # att_loss_seed = ...
        # joint_att_loss = att_loss_max + self.alpha_p * att_loss_seed

        # 5. 总联合损失
        total_joint_loss = joint_ctc_loss  # 可以加上 joint_att_loss

        return {
            "loss": total_joint_loss,
            "loss_max": ctc_loss_max,
            "loss_seed": ctc_loss_seed,
            "joint_ctc_loss": joint_ctc_loss
        }