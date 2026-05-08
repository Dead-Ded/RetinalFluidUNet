from .training import train_model
from .initialisation import *

if __name__ == "__main__":
    # Перенос на устройство (GPU или CPU)
    if device == torch.device("cpu"):
        print("No GPU available, using CPU.")
        raise NotImplementedError("No GPU available, using CPU.")

    print(aug_size)

    train_main = False
    train_rpe = False

    if train_main:
        # Обучение
        train_model(
            model=model_main,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=400,
            is_ped=False,
            num_classes=2
            # val_loader_vis=val_irf_srf_loader_for_vis
        )

    if train_rpe:
        # Обучение
        train_model(
            model=model_rpe,
            train_loader=rpe_train_loader,
            val_loader=rpe_val_loader,
            criterion=criterion,
            scheduler=scheduler,
            optimizer=optimizer,
            device=device,
            num_epochs=400,
            is_ped=True,
            # val_loader_vis=val_ped_loader_for_vis,
            num_classes=1
        )
