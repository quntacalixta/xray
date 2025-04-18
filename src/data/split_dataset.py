import os
import shutil
import random
from pathlib import Path

def augment_validation_set(data_dir='data/raw/chest_xray', val_percentage=0.15, seed=42):
    """
    Augmente l'ensemble de validation en déplaçant une partie des images d'entraînement
    
    Args:
        data_dir: Chemin vers le répertoire des données
        val_percentage: Pourcentage de l'ensemble d'entraînement à utiliser pour la validation
        seed: Graine aléatoire pour la reproductibilité
    """
    random.seed(seed)
    
    # Chemins des répertoires
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    
    # Créer une sauvegarde de l'ensemble de validation actuel
    backup_dir = Path(data_dir) / 'val_backup'
    if not backup_dir.exists() and val_dir.exists():
        print(f"Création d'une sauvegarde de l'ensemble de validation actuel dans {backup_dir}")
        shutil.copytree(val_dir, backup_dir)
    
    # Pour chaque classe (NORMAL, PNEUMONIA)
    for class_name in ['NORMAL', 'PNEUMONIA']:
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        # S'assurer que le répertoire de validation existe
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtenir toutes les images d'entraînement
        train_images = list(train_class_dir.glob('*.jpeg'))
        
        # Calculer le nombre d'images à déplacer
        num_to_move = int(len(train_images) * val_percentage)
        print(f"Déplacement de {num_to_move} images de la classe {class_name} vers l'ensemble de validation")
        
        # Sélectionner aléatoirement les images à déplacer
        images_to_move = random.sample(train_images, num_to_move)
        
        # Déplacer les images
        for img_path in images_to_move:
            # Créer le chemin de destination
            dest_path = val_class_dir / img_path.name
            
            # Vérifier si l'image existe déjà dans l'ensemble de validation
            if not dest_path.exists():
                shutil.move(str(img_path), str(dest_path))
    
    # Compter et afficher les statistiques
    count_images(data_dir)

def count_images(data_dir):
    """Compte et affiche le nombre d'images dans chaque ensemble et classe"""
    result = {}
    
    for split in ['train', 'val', 'test']:
        result[split] = {}
        split_dir = Path(data_dir) / split
        
        if not split_dir.exists():
            continue
            
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                continue
                
            n_images = len(list(class_dir.glob('*.jpeg')))
            result[split][class_name] = n_images
    
    print("\nDistribution des images après redistribution:")
    for split, classes in result.items():
        print(f"  {split.upper()}:")
        for class_name, count in classes.items():
            print(f"    {class_name}: {count} images")

if __name__ == "__main__":
    augment_validation_set()