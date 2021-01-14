from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()
    features = input.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())
    return G.div(batch_size * h * w * f_map_num)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()

        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class StyleTransferer():
    def __init__(self, content_layers, style_layers, cnn, device):
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.cnn = cnn
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.device = device

    def prepare_model(self, style_img, content_img):
        cnn = copy.deepcopy(self.cnn)
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)
        self.input_img = content_img.clone()

        self.content_losses = []
        self.style_losses = []

        self.model = nn.Sequential(normalization)

        i = 0  # counting convolution layers
        j = 1  # counting pooling layers
        for layer in cnn.children():

            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv{}-{}'.format(j, i)

            elif isinstance(layer, nn.ReLU):
                name = 'relu{}-{}'.format(j, i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool-{}'.format(j)
                j += 1
                i = 0

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn{}-{}'.format(j, i)
            else:
                raise RuntimeError('Not a standart layer found: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in self.content_layers:
                target = self.model(content_img).detach()
                content_loss = ContentLoss(target)
                self.model.add_module("content_loss_{}-{}".format(j, i), content_loss)
                self.content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = self.model(style_img).detach()

                style_loss = StyleLoss(target_feature)
                self.model.add_module("style_loss_{}-{}".format(j, i), style_loss)
                self.style_losses.append(style_loss)

        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)]

    def transfer(self, content_img, style_img, num_steps=500,
                 style_weight=10000, content_weight=1):

        self.prepare_model(style_img, content_img)

        optimizer = optim.LBFGS([self.input_img.requires_grad_()])

        iter = 0
        while iter <= num_steps:

            def closure():

                self.input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                self.model(self.input_img)

                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                return style_score + content_score

            iter += 1
            if iter % 100 == 0:
                print("Iteration {}".format(iter))

                optimizer.step(closure)

        self.input_img.data.clamp_(0, 1)

        return self.input_img


def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])

    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

transfer = None

#def start():


async def run(content_img, style_img, id):
    content_layers = ['conv1-1', 'conv2-1', 'conv3-1', 'conv4-1', 'conv5-1']
    style_layers = ['conv1-1', 'conv2-1', 'conv3-1', 'conv4-1', 'conv5-1']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.vgg16(pretrained=True).features.to(device).eval()

    transfer = StyleTransferer(content_layers, style_layers, cnn, device)

    content_img = image_loader(content_img, 128, device)
    style_img = image_loader(style_img, 128, device)

    unloader = transforms.ToPILImage()
    tensor = transfer.transfer(content_img, style_img, num_steps=500,
                             style_weight=10000, content_weight=1)
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    file = 'result_{}.jpg'.format(id)
    image.save(file)
