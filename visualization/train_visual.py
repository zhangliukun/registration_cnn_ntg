import torch.nn.functional as F
import torch
import visdom

from tnf_transform.img_process import normalize_image
import numpy as np

class VisdomHelper:
    def __init__(self,env_name):
        self.env_name = env_name
        self.vis = visdom.Visdom(env = self.env_name)

    def drawImage(self, source_image_batch, warped_image_batch, target_image_batch,single_channel = True,show_size=8):
        if source_image_batch.shape[0] > show_size:
            source_image_batch =source_image_batch[0:show_size]
            warped_image_batch =warped_image_batch[0:show_size]
            target_image_batch =target_image_batch[0:show_size]

        source_image_batch = normalize_image(source_image_batch,forward=False)
        warped_image_batch = normalize_image(warped_image_batch, forward=False)
        target_image_batch = normalize_image(target_image_batch, forward=False)

        source_image_batch = torch.clamp(source_image_batch,0,1)
        warped_image_batch = torch.clamp(warped_image_batch,0,1)
        target_image_batch = torch.clamp(target_image_batch,0,1)

        if single_channel:
            overlayImage = torch.cat((warped_image_batch, target_image_batch, warped_image_batch), 1)
        else:
            overlayImage = torch.cat((warped_image_batch[:,0:1,:,:],target_image_batch[:,1:2,:,:],warped_image_batch[:,0:1,:,:]),1)

        self.vis.images(
            overlayImage, win="overlay",
            opts=dict(title='overlay_image', caption='overlay.', width=1400, height=150, jpgquality=40)
        )

        self.vis.images(
            source_image_batch, win="s1",
            opts=dict(title='source_image_batch', caption='source.', width=1400, height=150, jpgquality=40)
        )
        self.vis.images(
            warped_image_batch, win="s2",
            opts=dict(title='warped_image_batch', caption='warped.', width=1400, height=150, jpgquality=40)
        )

        self.vis.images(
            target_image_batch, win="s3",
            opts=dict(title='target_image_batch', caption='target.', width=1400, height=150, jpgquality=40)
        )

    def drawLoss(self,epoch,train_loss):
        layout = dict(title="train_loss", xaxis={'title': 'epoch'}, yaxis={'title': 'train loss'})
        self.vis.line(X=torch.IntTensor([epoch]), Y=torch.FloatTensor([train_loss]), win="lineloss",
                 update='new' if epoch == 0 else 'append', opts=layout)

    def drawBothLoss(self,epoch,train_loss,test_loss,layout_title,x_axis = 'epoch',y_axis = 'loss',win='win'):
        layout = dict(title=layout_title, xaxis={'title': x_axis}, yaxis={'title': y_axis},legend=['train_loss','test_loss'])
        print(epoch, train_loss,test_loss)
        self.vis.line(X=np.column_stack((epoch,epoch)), Y=np.column_stack((train_loss,test_loss)), win=win,
                 update='new' if epoch == 0 else 'append', opts=layout)

    def drawGridlossGroup(self,X_list,ntg_list,Y_list_A,Y_list_B,Y_list_cvpr,
                          layout_title,x_axis ='grid_loss',y_axis ='num',
                          win='result',update='append'):
        layout = dict(title=layout_title, xaxis={'title': x_axis}, yaxis={'title': y_axis},
                      legend=['ntg_grid_loss','cnn_grid_loss','cnn_ntg_grid_loss','cvpr_2018_loss'],xlabel=X_list)
        self.vis.line(X=np.column_stack((X_list, X_list, X_list,X_list)),
                      Y=np.column_stack((ntg_list,Y_list_A, Y_list_B,Y_list_cvpr)),
                      update=update,
                      win=win,opts=layout)

    def drawGridlossBar(self,X_list,ntg_list,cnn_list_A,cnn_ntg_list_B,cvpr_list,layout_title,xlabel ='grid_loss(pixel)',ylabel ='percentage(%)',
                          win='result',update='append'):

        layout = dict(title=layout_title, xlabel=xlabel, ylabel=ylabel,
                      legend=['ntg_grid_loss', 'cnn_grid_loss', 'cnn_ntg_grid_loss', 'cvpr_2018_loss'], xaxis=X_list,stacked=False,
                        update=update,win=win,xtickstep=1)

        self.vis.bar(
            X=np.column_stack((ntg_list, cnn_list_A, cnn_ntg_list_B, cvpr_list)),opts=layout)



    def getVisdom(self):
        return self.vis

    def show_cnn_result(self,source_image_batch, warped_image_batch,fine_warped_image_batch, target_image_batch,single_channel = True):
        source_image_batch = normalize_image(source_image_batch, forward=False)
        warped_image_batch = normalize_image(warped_image_batch, forward=False)
        target_image_batch = normalize_image(target_image_batch, forward=False)

        source_image_batch = torch.clamp(source_image_batch, 0, 1)
        warped_image_batch = torch.clamp(warped_image_batch, 0, 1)
        target_image_batch = torch.clamp(target_image_batch, 0, 1)

        if single_channel:
            overlayImage = torch.cat((warped_image_batch, target_image_batch, warped_image_batch), 1)
        else:
            overlayImage = torch.cat(
                (warped_image_batch[:, 0:1, :, :], target_image_batch[:, 1:2, :, :], warped_image_batch[:, 0:1, :, :]),1)

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
            fine_warped_image_batch[0:8], win="s2_fine",
            opts=dict(title='fine_warped_image_batch', caption='warped.', width=1400, height=150, jpgquality=40)
        )

        self.vis.images(
            target_image_batch[0:8], win="s3",
            opts=dict(title='target_image_batch', caption='target.', width=1400, height=150, jpgquality=40)
        )


    def showImageBatch(self,image_batch,win='image',title='image',normailze=False,show_num=8,start_index = 0):
        if normailze:
            image_batch = normalize_image(image_batch, forward=False)
            image_batch = torch.clamp(image_batch, 0, 1)

        # self.vis.images(image_batch[0:show_num],win=win,opts=dict(title=title, caption='image_batch', width=1400, height=150, jpgquality=40))
        self.vis.images(image_batch[start_index:start_index+show_num],win=win,opts=dict(title=title, caption='image_batch', width=1400, height=150, jpgquality=40))

    def showHarvardBatch(self,image_batch,win='image',title='image',normailze=False,show_num=8,start_index = 17):
        if normailze:
            image_batch = normalize_image(image_batch, forward=False)
            image_batch = torch.clamp(image_batch, 0, 1)

        # self.vis.images(image_batch[0:show_num],win=win,opts=dict(title=title, caption='image_batch', width=1400, height=150, jpgquality=40))
        self.vis.images(image_batch[start_index:start_index+show_num],win=win,opts=dict(title=title, caption='image_batch', width=1400, height=150, jpgquality=40))

    #def draw_gridloss_group(self,x_list,y_list,win='table',title='table'):



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

