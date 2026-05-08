from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F


def analyze_efficientnet_structure():
    # Инициализируем модель
    model = EfficientNet.from_pretrained('efficientnet-b4')

    # Получаем информацию о блоках
    blocks_info = []

    # Информация о входном слое
    print(f"Stem: {model._conv_stem.out_channels} channels")

    for name, module in model._blocks._modules.items():
        print(name, module)
    # Информация о блоках
    for i, block in enumerate(model._blocks):
        input_channels = block._expand_conv.in_channels
        output_channels = block._project_conv.out_channels
        stride = block._depthwise_conv.stride[0]  # Получаем stride

        # Если stride > 1, это обычно означает понижение разрешения
        is_reduction = stride > 1

        blocks_info.append({
            'block_id': i,
            'input_ch': input_channels,
            'output_ch': output_channels,
            'reduction': is_reduction
        })

        print(f"Block {i}: Input={input_channels}, Output={output_channels}, Reduction={is_reduction}")

    # Информация о финальном слое
    print(f"Head: {model._conv_head.out_channels} channels")

    return blocks_info


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size in (3, 5)
        padding = 2 if kernel_size == 5 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        att_map = self.sigmoid(self.conv(x_cat))
        return x * att_map  # Элементное умножение на карту внимания


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=5):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        return self.relu(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Residual blocks вместо обычных сверток
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_channels // 2 + mid_channels),
            ResidualBlock(in_channels // 2 + mid_channels),
            nn.Dropout2d(0.1)
        )
        self.final_conv = nn.Conv2d(in_channels // 2 + mid_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        x = self.upsample(x)

        # Проверка и коррекция размеров
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)

        # Объединение с признаками из skip-connection
        x = torch.cat([x, skip], dim=1)
        x = self.res_blocks(x)
        return self.final_conv(x)


class EfficientPPM(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ) for size in [1, 2, 3, 6]
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        features = [x]
        for pool in self.pools:
            out = F.interpolate(pool(x), size=(h, w), mode='bilinear', align_corners=False)
            features.append(out)
        return self.project(torch.cat(features, dim=1))


class RetinalFluidUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b4')
        self.encoder_channels = [48, 32, 56, 112, 272]  # stem, block2, block6, block10, block22

        # Проекция skip-connections с уменьшением каналов и вниманием
        self.skip_proj = nn.ModuleList()
        for in_ch in self.encoder_channels:
            reduced_ch = max(in_ch // 2, 32)  # Уменьшаем каналы, но не менее 32
            self.skip_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, reduced_ch, 1),
                    nn.BatchNorm2d(reduced_ch),
                    nn.ReLU(inplace=True),
                    CBAM(reduced_ch)
                )
            )

        # PPM модуль
        self.ppm = EfficientPPM(1792, 512)

        # Декодер
        self.decoder4 = DecoderBlock(512, self.skip_proj[4][0].out_channels, 256)
        self.decoder3 = DecoderBlock(256, self.skip_proj[3][0].out_channels, 128)
        self.decoder2 = DecoderBlock(128, self.skip_proj[2][0].out_channels, 64)
        self.decoder1 = DecoderBlock(64, self.skip_proj[1][0].out_channels, 32)
        self.decoder0 = DecoderBlock(32, self.skip_proj[0][0].out_channels, 16)

        # Deep supervision
        self.ds_conv3 = nn.Conv2d(128, num_classes, 1)
        self.ds_conv1 = nn.Conv2d(32, num_classes, 1)

        # Финальная свертка
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'encoder' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def extract_encoder_features(self, x):
        features = []
        # Stem
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        features.append(x)  # 48 channels

        # Blocks
        blocks_idx = [2, 6, 10, 22]  # Индексы блоков для skip-connections
        for i, block in enumerate(self.encoder._blocks):
            x = block(x)
            if i in blocks_idx:
                features.append(x)

        # Head
        x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
        return x, features

    def forward(self, x):
        input_shape = x.shape[2:]

        # Получаем выходы энкодера
        encoder_output, skips = self.extract_encoder_features(x)

        # Проверка соответствия количества skip-connections
        if len(skips) != len(self.skip_proj):
            raise ValueError(f"Expected {len(self.skip_proj)} skip connections, got {len(skips)}!!!")

        # Обрабатываем skip-connections
        processed_skips = []
        for i, skip in enumerate(skips):
            # Проверка соответствия каналов
            if skip.size(1) != self.encoder_channels[i]:
                raise ValueError(
                    f"Skip connection {i} has {skip.size(1)} channels, expected {self.encoder_channels[i]}")
            processed_skips.append(self.skip_proj[i](skip))

        # Применяем PPM
        ppm_out = self.ppm(encoder_output)

        # Декодируем с deep supervision
        x4 = self.decoder4(ppm_out, processed_skips[4])
        x3 = self.decoder3(x4, processed_skips[3])
        ds3 = self.ds_conv3(x3)

        x2 = self.decoder2(x3, processed_skips[2])
        x1 = self.decoder1(x2, processed_skips[1])
        ds1 = self.ds_conv1(x1)

        x0 = self.decoder0(x1, processed_skips[0])
        main_output = self.final_conv(x0)

        # Ресайз до исходного размера
        if x0.shape[2:] != input_shape:
            main_output = F.interpolate(main_output, size=input_shape, mode='bilinear', align_corners=False)
            ds3 = F.interpolate(ds3, size=input_shape, mode='bilinear', align_corners=False)
            ds1 = F.interpolate(ds1, size=input_shape, mode='bilinear', align_corners=False)

        if self.training:
            return main_output, ds3, ds1

        return main_output


def predict_combined(image, models: list):
    preds = [
        torch.sigmoid(models[i](image)) for i in range(len(models))
    ]

    # Объединение результатов
    final_mask = torch.cat([
        *preds
    ], dim=1)

    return final_mask > 0.5


def load_model(model_path: str, num_classes: int):
    model = RetinalFluidUNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


class CombinedSegmentationModel(nn.Module):
    """
    Обёртка для нескольких моделей, каждая из которых предсказывает маску определённого класса.
    На выходе возвращает тензор [B, num_classes, H, W] с вероятностями (или логитами).
    """
    def __init__(self, class_configs, device='cuda'):
        """
        class_configs: словарь вида
            {
                'IRF': {
                    'model': model_irf,          # объект nn.Module
                    'channel': 0,                # индекс канала в выходе модели (None если одноканальный)
                    'activation': 'sigmoid',     # 'sigmoid', 'softmax', или None (уже вероятности)
                },
                'SRF': {
                    'model': model_main,
                    'channel': 1,
                    'activation': 'sigmoid',
                },
                'PED': {
                    'model': model_rpe,
                    'channel': None,
                    'activation': 'sigmoid',
                },
                ...
            }
        device: устройство для выполнения (модели уже должны быть на нужном устройстве)
        """
        super().__init__()
        self.class_configs = class_configs
        self.num_classes = len(class_configs)
        self.device = device

        # Регистрируем модели как подмодули (чтобы .to(device) и .eval() работали централизованно)
        for class_name, cfg in class_configs.items():
            setattr(self, f"model_{class_name}", cfg['model'])

    def forward(self, x):
        """
        x: тензор [B, 3, H, W] (уже нормализованный и приведённый к нужному размеру)
        Возвращает: тензор [B, num_classes, H, W] с вероятностями
        """
        B, _, H, W = x.shape
        outputs = []

        for class_name, cfg in self.class_configs.items():
            model = cfg['model']
            model.eval()
            with torch.no_grad():
                out = model(x)  # форма выхода зависит от модели

            # Извлекаем нужный канал, если указан
            if cfg.get('channel') is not None:
                out = out[:, cfg['channel'], :, :].unsqueeze(1)  # [B, 1, H, W]
            # Если out многоканальный (C>1) и channel не указан, возможно, нужно применить argmax?
            # Для простоты предполагаем, что модель выдаёт одноканальный выход или нужный канал уже выбран.

            # Применяем активацию для получения вероятностей
            act = cfg.get('activation', 'sigmoid')
            if act == 'sigmoid':
                prob = torch.sigmoid(out)  # [B, 1, H, W]
            elif act == 'softmax':
                prob = F.softmax(out, dim=1)  # [B, C, H, W] -> берём все каналы? Лучше уточнить.
                # Если softmax на несколько классов, то нужно заранее определить маппинг
                raise NotImplementedError("Softmax активация требует отдельной обработки маппинга каналов")
            else:
                prob = out  # предполагаем, что это уже вероятности

            outputs.append(prob)

        # Объединяем по каналам: [B, num_classes, H, W]
        combined = torch.cat(outputs, dim=1)
        return combined
