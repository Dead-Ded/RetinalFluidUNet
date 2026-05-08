# Функция для постепенного размораживания энкодера
from itertools import chain


def unfreeze_encoder_layers(model, optimizer, epoch):
    """
    Точная разморозка блоков EfficientNet-B4 на основе их структуры
    """
    # Получаем общее количество блоков в энкодере
    total_blocks = len(model.encoder._blocks)

    # Стратегия разморозки по эпохам
    if epoch == 15:
        # Размораживаем поздние блоки (последние 10) + head
        start_idx = max(0, total_blocks - 10)
        for i in range(start_idx, total_blocks):
            for param in model.encoder._blocks[i].parameters():
                param.requires_grad = True

        # Размораживаем head
        for param in chain(model.encoder._conv_head.parameters(),
                          model.encoder._bn1.parameters()):
            param.requires_grad = True

        # Устанавливаем LR для поздних слоев (только если группа существует)
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'encoder':
                param_group['lr'] = 1e-5
                print(f"Разморожены блоки {start_idx}-{total_blocks-1} и head, LR установлен на 1e-5")
                break
        else:
            print("Предупреждение: группа параметров 'encoder' не найдена в оптимизаторе")

    elif epoch == 30:
        # Размораживаем средние блоки (с 10 до total_blocks-10)
        start_idx = max(0, 10)
        end_idx = max(start_idx + 1, total_blocks - 10)  # Убедимся, что end_idx > start_idx
        for i in range(start_idx, end_idx):
            for param in model.encoder._blocks[i].parameters():
                param.requires_grad = True

        # Увеличиваем LR для средних слоев (только если группа существует)
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'encoder':
                param_group['lr'] = 3e-5
                print(f"Разморожены блоки {start_idx}-{end_idx-1}, LR установлен на 3e-5")
                break
        else:
            print("Предупреждение: группа параметров 'encoder' не найдена в оптимизаторе")

    elif epoch == 45:
        # Размораживаем ранние блоки (первые 10) + stem
        end_idx = min(10, total_blocks)
        for i in range(0, end_idx):
            for param in model.encoder._blocks[i].parameters():
                param.requires_grad = True

        # Размораживаем stem
        for param in chain(model.encoder._conv_stem.parameters(),
                          model.encoder._bn0.parameters()):
            param.requires_grad = True

        # Увеличиваем LR для ранних слоев (только если группа существует)
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'encoder':
                param_group['lr'] = 1e-4
                print(f"Разморожены блоки 0-{end_idx-1} и stem, LR установлен на 1e-4")
                break
        else:
            print("Предупреждение: группа параметров 'encoder' не найдена в оптимизаторе")

def check_freezing_status(model):
    """Проверяет статус заморозки параметров модели"""
    total_params = 0
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
        else:
            frozen_params += 1

    print(f"Общая статистика: {frozen_params}/{total_params} параметров заморожено ({frozen_params/total_params*100:.1f}%)")

    # Детальная информация по энкодеру
    encoder_frozen = 0
    encoder_total = 0
    encoder_trainable = 0

    for name, param in model.encoder.named_parameters():
        encoder_total += 1
        if param.requires_grad:
            encoder_trainable += 1
        else:
            encoder_frozen += 1

    print(f"Энкодер: {encoder_frozen}/{encoder_total} замороженных параметров ({encoder_frozen/encoder_total*100:.1f}%)")
    print(f"Энкодер: {encoder_trainable}/{encoder_total} обучаемых параметров ({encoder_trainable/encoder_total*100:.1f}%)")

# Learning rate warmup
def adjust_learning_rate(optimizer, epoch, warmup_epochs=10, initial_lr=3e-4):
    """
    Learning rate warmup только для декодера
    """
    if epoch < warmup_epochs:
        # Линейное увеличение LR в течение warmup_epochs только для декодера
        lr = initial_lr * (epoch + 1) / warmup_epochs

        # Ищем группу параметров декодера
        decoder_group = None
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'decoder':
                decoder_group = param_group
                break

        if decoder_group:
            decoder_group['lr'] = lr
            print(f"Warmup: установлен LR декодера на {lr:.2e}")
        else:
            print("Предупреждение: группа параметров 'decoder' не найдена в оптимизаторе")
