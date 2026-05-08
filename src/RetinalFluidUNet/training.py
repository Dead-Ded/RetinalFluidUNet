import gc
import time

from tqdm import tqdm

from .variables import *
from .metrics import *
from .visualisation import *
from .learning_control import *
from losses import *
from .models import *


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def create_log_file(model_name, is_ped, num_classes):
    """Создает файл для логирования с уникальным именем"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_type = "PED" if is_ped else "IRF_SRF"
    log_file_name = f"{model_name}_{timestamp}_log.csv" if model_name else f"{model_type}_{timestamp}_log.csv"
    log_file_path = os.path.join(SAVE_MODEL_PATH, log_file_name)

    # Создаем заголовок CSV
    columns = [
        "Epoch", "Train_Loss", "Val_Loss",
        "Train_Dice", "Val_Dice",
        "Train_Precision", "Val_Precision",
        "Train_Recall", "Val_Recall",
        "LR"
    ]

    for i in range(num_classes):
        for metric in ['dice', 'precision', 'recall']:
            columns.append(f"Train_{metric}_class_{i}")
            columns.append(f"Val_{metric}_class_{i}")

    with open(log_file_path, "w") as f:
        f.write(",".join(columns) + "\n")

    return log_file_path


def log_epoch_results(log_file_path, epoch, train_metrics, val_metrics,
                      train_class_metrics, val_class_metrics, current_lr, num_classes):
    """Записывает результаты эпохи в лог-файл"""
    values = [
        str(epoch + 1),
        f"{train_metrics['loss']:.4f}",
        f"{val_metrics['loss']:.4f}",
        f"{train_metrics['dice']:.4f}",
        f"{val_metrics['dice']:.4f}",
        f"{train_metrics['precision']:.4f}",
        f"{val_metrics['precision']:.4f}",
        f"{train_metrics['recall']:.4f}",
        f"{val_metrics['recall']:.4f}",
        f"{current_lr:.8f}"
    ]

    for i in range(num_classes):
        for metric in ['dice', 'precision', 'recall']:
            train_val = train_class_metrics[f'{metric}_class_{i}'][-1] if train_class_metrics[
                f'{metric}_class_{i}'] else 0.0
            val_val = val_class_metrics[f'{metric}_class_{i}'][-1] if val_class_metrics[f'{metric}_class_{i}'] else 0.0
            values.append(f"{train_val:.4f}")
            values.append(f"{val_val:.4f}")

    with open(log_file_path, "a") as f:
        f.write(",".join(values) + "\n")


def run_epoch(model, data_loader, criterion, device, num_classes, mode='train', optimizer=None):
    """
    Выполняет одну эпоху обучения или валидации.

    Args:
        model: PyTorch модель
        data_loader: DataLoader
        criterion: Функция потерь
        device: Устройство (cpu/cuda)
        num_classes: Количество классов
        mode: 'train' или 'val'
        optimizer: Оптимизатор (обязателен при mode='train')

    Returns:
        avg_metrics: Словарь с усреднёнными метриками эпохи
        class_metrics: Словарь с метриками по каждому классу
    """
    is_training = mode == 'train'
    if is_training:
        model.train()
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training mode.")
    else:
        model.eval()

    # Инициализация аккумуляторов
    running_loss = 0.0
    running_dice = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_class_metrics = {
        f'{metric}_class_{i}': 0.0
        for i in range(num_classes) for metric in ['dice', 'precision', 'recall']
    }
    dice_scores = [] if not is_training else None

    desc = 'Training' if is_training else 'Validation'
    epoch_iter = tqdm(data_loader, desc=desc, leave=False)

    for images, masks in epoch_iter:
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()
        outputs = None
        loss = None

        if is_training:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, masks)

        # Извлечение основного выхода (поддержка auxiliary outputs)
        main_output = outputs[0] if isinstance(outputs, tuple) else outputs

        # Вычисление метрик без градиентов (экономит память и ускоряет расчёт)
        with torch.no_grad():
            batch_dice = dice_coef(main_output, masks)
            batch_precision = precision_coef(main_output, masks)
            batch_recall = recall_coef(main_output, masks)

            running_loss += loss.item()
            running_dice += batch_dice
            running_precision += batch_precision
            running_recall += batch_recall

            if dice_scores is not None:
                dice_scores.append(batch_dice.item() if isinstance(batch_dice, torch.Tensor) else batch_dice)

            class_metrics = calculate_metrics_per_class(outputs, masks, num_classes=num_classes)
            for key in running_class_metrics:
                running_class_metrics[key] += class_metrics[key]

        epoch_iter.set_postfix(loss=loss.item())

    num_batches = len(data_loader)
    if num_batches == 0:
        empty_metrics = {'loss': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        if dice_scores is not None:
            empty_metrics['dice_scores'] = []
        return empty_metrics, running_class_metrics

    # Усреднение метрик
    avg_metrics = {
        'loss': running_loss / num_batches,
        'dice': running_dice / num_batches,
        'precision': running_precision / num_batches,
        'recall': running_recall / num_batches
    }

    if dice_scores is not None:
        avg_metrics['dice_scores'] = dice_scores

    for key in running_class_metrics:
        running_class_metrics[key] /= num_batches

    return avg_metrics, running_class_metrics


def train_model(
        model: RetinalFluidUNet, train_loader: DataLoader, val_loader: DataLoader,
        criterion, optimizer, scheduler, device, num_epochs, is_ped: bool,
        file_name: str = None, num_classes: int = 2,
):
    # Инициализация метрик
    train_metrics_list = []
    val_metrics_list = []
    train_class_metrics = {f'{metric}_class_{i}': [] for i in range(num_classes) for metric in
                           ['dice', 'precision', 'recall']}
    val_class_metrics = {f'{metric}_class_{i}': [] for i in range(num_classes) for metric in
                         ['dice', 'precision', 'recall']}

    model.to(device)
    start_time = time.time()
    best_loss = torch.inf

    # Создаем уникальный файл для логирования
    log_file_path = create_log_file(file_name, is_ped, num_classes)

    early_stopper = EarlyStopping(patience=20)
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # Проверка памяти и статуса заморозки
        if epoch % 10 == 0:
            check_freezing_status(model)
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
                memory_cached = torch.cuda.memory_reserved(device) / 1024 ** 3
                print(f"Эпоха {epoch}: {memory_allocated:.2f} GB / {memory_cached:.2f} GB")
                if memory_cached > 7.0:
                    print("Очистка памяти...")
                    torch.cuda.empty_cache()
                    gc.collect()

        # Размораживание энкодера и регулировка LR
        unfreeze_encoder_layers(model, optimizer, epoch)
        adjust_learning_rate(optimizer, epoch)

        # Визуализация каждые 5 эпох
        if epoch % 5 == 0:
            visualize_predictions(model, val_loader, device, class_names=None, num_examples=3)

        # Обучение
        train_metrics, train_class_metrics_epoch = run_epoch(
            model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device,
            num_classes=num_classes, mode="train"
        )

        # Сохраняем метрики обучения
        train_metrics_list.append(train_metrics)
        for key in train_class_metrics:
            train_class_metrics[key].append(train_class_metrics_epoch[key])

        # Валидация
        val_metrics, val_class_metrics_epoch = run_epoch(
            model=model, data_loader=val_loader, criterion=criterion, device=device, num_classes=num_classes,
            mode="validation"
        )

        # Сохраняем метрики валидации
        val_metrics_list.append(val_metrics)
        for key in val_class_metrics:
            val_class_metrics[key].append(val_class_metrics_epoch[key])

        # Обновление scheduler
        if epoch >= 10:
            scheduler.step(val_metrics['loss'])

        # Сохранение лучшей модели
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            model_file_name = file_name + ".pth" if file_name else ("PED.pth" if is_ped else 'IRF_SRF.pth')
            torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, model_file_name))
            print(f"Epoch {epoch + 1}: Лучшая модель сохранена с loss={best_loss:.4f}")

        # Логирование и вывод информации
        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Логирование в файл
        log_epoch_results(
            log_file_path, epoch, train_metrics, val_metrics,
            train_class_metrics, val_class_metrics, current_lr, num_classes
        )

        # Формируем строку с метриками по классам для вывода в консоль
        class_metrics_str = ""
        for i in range(num_classes):
            class_metrics_str += f" | Class {i}: D={val_class_metrics_epoch[f'dice_class_{i}']:.3f}, P={val_class_metrics_epoch[f'precision_class_{i}']:.3f}, R={val_class_metrics_epoch[f'recall_class_{i}']:.3f}"

        # Вывод подробной информации в консоль
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Dice: {np.mean(val_metrics['dice_scores']):.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}"
              f"{class_metrics_str} | "
              f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} "
              f"... | LR: {current_lr:.8f}")

        # Обновление прогресс-бара
        epoch_pbar.set_postfix({
            'train_loss': f"{train_metrics['loss']:.4f}",
            'val_loss': f"{val_metrics['loss']:.4f}",
            'val_dice': f"{np.mean(val_metrics['dice_scores']):.4f}",
            'val_precision': f"{val_metrics['precision']:.4f}",
            'val_recall': f"{val_metrics['recall']:.4f}",
            'lr': f"{current_lr:.8f}"
        })

        # Проверка ранней остановки
        early_stopper(val_metrics['loss'])
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    # Завершение обучения
    epoch_pbar.close()
    print("Training completed.")
    print(f"Общее время обучения: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))} мин.")
    print(f"Лучшая модель сохранена с loss={best_loss:.4f}")

    # Визуализация результатов
    paint_it_Black(
        num_epochs,
        [m['loss'] for m in train_metrics_list],
        [m['loss'] for m in val_metrics_list],
        [m['dice'] for m in train_metrics_list],
        [m['dice'] for m in val_metrics_list],
        [m['precision'] for m in train_metrics_list],
        [m['precision'] for m in val_metrics_list],
        [m['recall'] for m in train_metrics_list],
        [m['recall'] for m in val_metrics_list]
    )

    # Сохранение финальной модели
    final_model_path = os.path.join(SAVE_MODEL_PATH, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    return train_metrics_list, val_metrics_list, train_class_metrics, val_class_metrics


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list = None,
    weights_path: str = None,
    threshold: float = 0.5,
    save_vis_dir: str = None,
    num_vis_examples: int = 5,
    return_predictions: bool = False
):
    """
    Оценка модели на тестовой выборке.

    Args:
        model: Экземпляр модели (RetinalFluidUNet).
        test_loader: DataLoader с тестовыми данными.
        device: Устройство (cuda/cpu).
        num_classes: Количество классов сегментации.
        class_names: Список имён классов (опционально).
        weights_path: Путь к файлу .pth с сохранёнными весами модели.
                      Если None, используется текущее состояние model.
        threshold: Порог бинаризации для вероятностей (по умолчанию 0.5).
        save_vis_dir: Директория для сохранения визуализаций (если None, визуализация не выполняется).
        num_vis_examples: Количество примеров для визуализации.
        return_predictions: Если True, возвращает список предсказаний и масок.

    Returns:
        dict: Словарь со средними метриками по тесту.
        (опционально) tuple: (metrics, all_preds, all_masks) при return_predictions=True.
    """
    # Загрузка весов, если указан путь
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Веса модели загружены из {weights_path}")

    model.to(device)
    model.eval()

    # Инициализация аккумуляторов метрик
    total_loss = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    class_metrics = {f'{metric}_class_{i}': 0.0 for i in range(num_classes)
                     for metric in ['dice', 'precision', 'recall']}

    all_preds = [] if return_predictions else None
    all_masks = [] if return_predictions else None

    # Критерий для вычисления loss (используем только для информации)
    dice_loss_fn = DiceLoss()
    focal_loss_fn = FocalLoss()

    vis_counter = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device).float()

            # Forward pass
            outputs = model(images)

            # Если модель возвращает кортеж (deep supervision), берём основной выход
            main_output = outputs[0] if isinstance(outputs, tuple) else outputs

            # Вычисление loss (для информации)
            loss = 0.5 * focal_loss_fn(main_output, masks) + 0.5 * dice_loss_fn(main_output, masks)
            total_loss += loss.item()

            # Метрики по всему батчу
            batch_dice = dice_coef(main_output, masks, thresholds=[threshold] * num_classes)
            batch_precision = precision_coef(main_output, masks, threshold=threshold)
            batch_recall = recall_coef(main_output, masks, threshold=threshold)

            total_dice += batch_dice
            total_precision += batch_precision
            total_recall += batch_recall

            # Метрики по классам
            batch_class_metrics = calculate_metrics_per_class(main_output, masks, num_classes)
            for key in class_metrics:
                class_metrics[key] += batch_class_metrics[key]

            # Сохранение предсказаний, если нужно
            if return_predictions:
                probs = torch.sigmoid(main_output)
                preds = (probs > threshold).float()
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())

            # Визуализация (если задана директория и ещё не достигнут лимит)
            if save_vis_dir is not None and vis_counter < num_vis_examples:
                probs = torch.sigmoid(main_output)
                preds = (probs > threshold).float()
                for i in range(images.size(0)):
                    if vis_counter >= num_vis_examples:
                        break
                    # Используем существующую функцию визуализации
                    visualisation(images[i].cpu(), masks[i].cpu(), preds[i].cpu(),
                                  class_names=class_names)
                    os.makedirs(save_vis_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_vis_dir, f"test_vis_{vis_counter}.png"),
                                bbox_inches='tight')
                    plt.close()
                    vis_counter += 1

    # Усреднение метрик
    num_batches = len(test_loader)
    avg_metrics = {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches
    }

    for key in class_metrics:
        class_metrics[key] /= num_batches

    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*50)
    print(f"Loss:      {avg_metrics['loss']:.4f}")
    print(f"Dice (avg): {avg_metrics['dice']:.4f}")
    print(f"Precision:  {avg_metrics['precision']:.4f}")
    print(f"Recall:     {avg_metrics['recall']:.4f}")
    print("\nМетрики по классам:")
    for i in range(num_classes):
        name = class_names[i] if class_names else f"Class {i}"
        d = class_metrics[f'dice_class_{i}']
        p = class_metrics[f'precision_class_{i}']
        r = class_metrics[f'recall_class_{i}']
        print(f"  {name:12s}: Dice={d:.4f}, Precision={p:.4f}, Recall={r:.4f}")
    print("="*50)

    if return_predictions:
        # Конкатенация всех батчей
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        return avg_metrics, class_metrics, all_preds, all_masks
    else:
        return avg_metrics, class_metrics
