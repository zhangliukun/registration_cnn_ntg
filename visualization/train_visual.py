import torch.nn.functional as F
import torch
import visdom

from tnf_transform.img_process import normalize_image


class VisdomHelper:
    def __init__(self,env_name):
        self.env_name = env_name
        self.vis = visdom.Visdom(env = self.env_name)

    def drawImage(self, source_image_batch, warped_image_batch, target_image_batch):
        source_image_batch = normalize_image(source_image_batch,forward=False)
        warped_image_batch = normalize_image(warped_image_batch, forward=False)
        target_image_batch = normalize_image(target_image_batch, forward=False)

        source_image_batch = torch.clamp(source_image_batch,0,1)
        warped_image_batch = torch.clamp(warped_image_batch,0,1)
        target_image_batch = torch.clamp(target_image_batch,0,1)

        overlayImage = torch.cat((warped_image_batch, target_image_batch, warped_image_batch), 1)

        self.vis.images(
            overlayImage[0:8], win="overlay",
            opts=dict(title='overlay_image', caption='overlay.', width=1400, height=150, jpgquality=40)
        )

        self.vis.images(
            source_image_batch[0:8], win="s1",
            opts=dict(title='source_image_batch', caption='source.', width=1400, height=150, jpgquality=40)
        )
        self.vis.images(
            warped_image_batch[0:8], win="s2",
            opts=dict(title='warped_image_batch', caption='warped.', width=1400, height=150, jpgquality=40)
        )

        self.vis.images(
            target_image_batch[0:8], win="s3",
            opts=dict(title='target_image_batch', caption='target.', width=1400, height=150, jpgquality=40)
        )

    def drawLoss(self,epoch,train_loss):
        layout = dict(title="train_loss", xaxis={'title': 'epoch'}, yaxis={'title': 'train loss'})
        self.vis.line(X=torch.IntTensor([epoch]), Y=torch.FloatTensor([train_loss]), win="lineloss",
                 update='new' if epoch == 0 else 'append', opts=layout)

    def show_cnn_result(self,source_image_batch, warped_image_batch, target_image_batch):
        source_image_batch = normalize_image(source_image_batch, forward=False)
        warped_image_batch = normalize_image(warped_image_batch, forward=False)
        target_image_batch = normalize_image(target_image_batch, forward=False)

        source_image_batch = torch.clamp(source_image_batch, 0, 1)
        warped_image_batch = torch.clamp(warped_image_batch, 0, 1)
        target_image_batch = torch.clamp(target_image_batch, 0, 1)

        overlayImage = torch.cat((warped_image_batch, target_image_batch, warped_image_batch), 1)

        self.vis.images(
            overlayImage[0:8], win="overlay",
            opts=dict(title='overlay_image', caption='overlay.', width=1400, height=150, jpgquality=40)
        )

        self.vis.images(
            source_image_batch[0:8], win="s1",
            opts=dict(title='source_image_batch', caption='source.', width=1400, height=150, jpgquality=40)
        )
        self.vis.images(
            warped_image_batch[0:8], win="s2",
            opts=dict(title='warped_image_batch', caption='warped.', width=1400, height=150, jpgquality=40)
        )

        self.vis.images(
            target_image_batch[0:8], win="s3",
            opts=dict(title='target_image_batch', caption='target.', width=1400, height=150, jpgquality=40)
        )

# def showImage(source_image_batch,warped_image_batch,target_image_batch,isShowRGB=True):
#
#     source_image_batch = source_image_batch.cpu().detach().numpy()
#     warped_image_batch = warped_image_batch.cpu().detach().numpy()
#     target_image_batch = target_image_batch.cpu().detach().numpy()
#     fig, axs = plt.subplots(3, 4)
#     for i in range(4):
#         image1 = np.transpose(source_image_batch[i], (1, 2, 0))
#         image2 = np.transpose(warped_image_batch[i], (1, 2, 0))
#         image3 = np.transpose(target_image_batch[i], (1, 2, 0))
#         if isShowRGB:
#             axs[0, i].imshow(image1)
#             axs[1, i].imshow(image2)
#             axs[2, i].imshow(image3)
#         else:
#             axs[0, i].imshow(image1.squeeze(),cmap='gray')
#             axs[1, i].imshow(image2.squeeze(),cmap='gray')
#             axs[2, i].imshow(image3.squeeze(),cmap='gray')
#     plt.show()

