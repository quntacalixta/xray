#!/usr/bin/env python3
"""
Script d'évaluation complète du modèle de classification de radiographies pulmonaires.
Génère des métriques détaillées et des visualisations pour le rapport.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    precision_recall_curve, auc, f1_score
)
from tqdm import tqdm

# Importer les modules du projet
from src.models.model import ChestXRayClassifier
from src.data.preprocessing import create_data_loaders
from src.utils.grad_cam import visualize_gradcam  # Si disponible

def calculate_clinical_metrics(y_true, y_pred):
    """
    Calculer des métriques cliniques pertinentes:
    - Sensibilité (Recall)
    - Spécificité
    - Valeur Prédictive Positive (PPV)
    - Valeur Prédictive Négative (NPV)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv
    }
    
    print("\nMétriques cliniques:")
    print(f"Sensibilité (Recall): {sensitivity:.4f}")
    print(f"Spécificité: {specificity:.4f}")
    print(f"Valeur Prédictive Positive (PPV): {ppv:.4f}")
    print(f"Valeur Prédictive Négative (NPV): {npv:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, output_dir="results"):
    """Créer et sauvegarder une matrice de confusion"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculer les pourcentages par ligne
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Créer une matrice avec valeurs absolues et pourcentages
    labels = np.array([
        [f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])
    
    ax = sns.heatmap(
        cm, annot=labels, fmt='', cmap='Blues',
        xticklabels=['NORMAL', 'PNEUMONIE'],
        yticklabels=['NORMAL', 'PNEUMONIE'],
        cbar=False
    )
    
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de confusion')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    print(f"Matrice de confusion sauvegardée dans {output_dir}/confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_prob, output_dir="results"):
    """Créer et sauvegarder une courbe ROC"""
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    print(f"Courbe ROC sauvegardée dans {output_dir}/roc_curve.png")
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, output_dir="results"):
    """Créer et sauvegarder une courbe Precision-Recall"""
    os.makedirs(output_dir, exist_ok=True)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    print(f"Courbe Precision-Recall sauvegardée dans {output_dir}/precision_recall_curve.png")
    plt.close()

def plot_threshold_analysis(y_true, y_prob, output_dir="results"):
    """Analyser l'impact du seuil de classification sur différentes métriques"""
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': []
    }
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculer les métriques
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['specificity'].append(specificity)
    
    # Trouver le meilleur seuil pour F1
    best_idx = np.argmax(metrics['f1'])
    best_threshold = thresholds[best_idx]
    best_f1 = metrics['f1'][best_idx]
    
    # Représentation graphique
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics['accuracy'], label='Accuracy')
    plt.plot(thresholds, metrics['precision'], label='Precision')
    plt.plot(thresholds, metrics['recall'], label='Recall')
    plt.plot(thresholds, metrics['f1'], label='F1', linewidth=2)
    plt.plot(thresholds, metrics['specificity'], label='Specificity')
    
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f'Seuil optimal = {best_threshold:.2f} (F1 = {best_f1:.3f})')
    
    plt.xlabel('Seuil de classification')
    plt.ylabel('Valeur métrique')
    plt.title('Impact du seuil de classification sur les métriques')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300)
    print(f"Analyse des seuils sauvegardée dans {output_dir}/threshold_analysis.png")
    plt.close()
    
    # Créer un tableau récapitulatif des performances à différents seuils
    key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_metrics = []
    
    for threshold in key_thresholds:
        idx = np.abs(thresholds - threshold).argmin()
        threshold_metrics.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'][idx],
            'sensitivity': metrics['recall'][idx],
            'specificity': metrics['specificity'][idx],
            'f1': metrics['f1'][idx]
        })
    
    # Marquer le seuil optimal
    threshold_df = pd.DataFrame(threshold_metrics)
    
    # Sauvegarder le tableau
    threshold_df.to_csv(os.path.join(output_dir, 'threshold_metrics.csv'), index=False)
    print(f"Métriques par seuil sauvegardées dans {output_dir}/threshold_metrics.csv")
    
    return best_threshold, best_f1

def plot_clinical_metrics(metrics, output_dir="results"):
    """Créer une visualisation des métriques cliniques"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Préparer les données
    metrics_to_plot = {
        'Sensibilité': metrics['sensitivity'] * 100,
        'Spécificité': metrics['specificity'] * 100,
        'VPP': metrics['ppv'] * 100,
        'VPN': metrics['npv'] * 100
    }
    
    # Créer les barres
    bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), 
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 100)
    plt.title('Métriques cliniques du modèle', fontsize=16)
    plt.ylabel('Pourcentage (%)')
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter des explications
    plt.figtext(0.5, 0.01, 
               "VPP: Valeur Prédictive Positive - Probabilité que le patient ait réellement une pneumonie quand le test est positif\n" + 
               "VPN: Valeur Prédictive Négative - Probabilité que le patient n'ait pas de pneumonie quand le test est négatif",
               ha='center', fontsize=9, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'clinical_metrics.png'), dpi=300)
    print(f"Métriques cliniques sauvegardées dans {output_dir}/clinical_metrics.png")
    plt.close()

def analyze_error_cases(model, data_loader, device, threshold, output_dir="results/error_cases"):
    """
    Analyser et visualiser les cas d'erreur (faux positifs et faux négatifs)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    all_images = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Collecter les prédictions
    print("Collecte des prédictions pour analyse d'erreurs...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds)
            all_probs.append(probs)
    
    # Convertir en arrays numpy
    all_images = torch.cat(all_images, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    # Identifier les faux positifs et faux négatifs
    false_positives = (all_labels == 0) & (all_preds == 1)
    false_negatives = (all_labels == 1) & (all_preds == 0)
    
    fp_count = np.sum(false_positives)
    fn_count = np.sum(false_negatives)
    
    print(f"Faux positifs trouvés: {fp_count}")
    print(f"Faux négatifs trouvés: {fn_count}")
    
    # Visualiser quelques cas d'erreur si disponibles
    if fp_count > 0 or fn_count > 0:
        # Fonction pour visualiser des échantillons d'erreurs
        def visualize_error_samples(error_mask, error_type, max_samples=6):
            if np.sum(error_mask) == 0:
                print(f"Aucun cas de {error_type} trouvé.")
                return
            
            indices = np.where(error_mask)[0]
            sample_count = min(max_samples, len(indices))
            
            # Sélectionner aléatoirement des échantillons
            if len(indices) > sample_count:
                np.random.seed(42)  # pour reproductibilité
                indices = np.random.choice(indices, sample_count, replace=False)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, idx in enumerate(indices):
                if i >= sample_count:
                    break
                    
                # Afficher l'image
                img = all_images[idx].transpose(1, 2, 0)
                # Dénormaliser l'image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"Prob: {all_probs[idx]:.2f}")
                axes[i].axis('off')
                
                # Si des axes supplémentaires existent (pour moins de 6 échantillons)
                for j in range(sample_count, len(axes)):
                    axes[j].axis('off')
            
            plt.suptitle(f"Échantillons de {error_type}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{error_type}.png"), dpi=300)
            print(f"Échantillons de {error_type} sauvegardés dans {output_dir}/{error_type}.png")
            plt.close()
        
        # Visualiser les faux positifs et faux négatifs
        print("Génération des visualisations d'erreurs...")
        visualize_error_samples(false_positives, "Faux_Positifs")
        visualize_error_samples(false_negatives, "Faux_Négatifs")

def evaluate(threshold=0.5, model_path='best_model.pth', data_dir='data/raw/chest_xray', output_dir='results', visualize=True):
    """
    Fonction principale pour l'évaluation complète du modèle avec visualisations
    
    Args:
        threshold (float): Seuil de classification
        model_path (str): Chemin vers le modèle
        data_dir (str): Répertoire des données
        output_dir (str): Répertoire pour les résultats et visualisations
        visualize (bool): Générer des visualisations
    
    Returns:
        dict: Résultats d'évaluation
    """
    # Créer les répertoires de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXRayClassifier.load_from_checkpoint(model_path, device=device)
    model.eval()
    
    print(f"Modèle chargé depuis {model_path}")
    print(f"Utilisation du périphérique: {device}")
    
    # Charger les données de test
    _, _, test_loader = create_data_loaders(data_dir, batch_size=32)
    print(f"Données de test chargées: {len(test_loader.dataset)} images")
    
    # Collecter les prédictions
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("Évaluation du modèle...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Obtenir les probabilités et prédictions
            probs = probabilities[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    # Convertir en numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculer et afficher les métriques principales avec le seuil fourni
    print("\nMatrice de confusion:")
    print(confusion_matrix(all_labels, all_preds))
    
    print(f"\nRapport de classification (seuil={threshold:.2f}):")
    print(classification_report(
        all_labels, all_preds,
        target_names=['NORMAL', 'PNEUMONIA']
    ))
    
    # Calculer les métriques cliniques
    clinical_metrics = calculate_clinical_metrics(all_labels, all_preds)
    
    # Générer les visualisations
    if visualize:
        print("\nGénération des visualisations...")
        
        # 1. Matrice de confusion
        plot_confusion_matrix(all_labels, all_preds, output_dir)
        
        # 2. Courbe ROC
        plot_roc_curve(all_labels, all_probs, output_dir)
        
        # 3. Courbe Precision-Recall
        plot_precision_recall_curve(all_labels, all_probs, output_dir)
        
        # 4. Analyse des seuils
        optimal_threshold, _ = plot_threshold_analysis(all_labels, all_probs, output_dir)
        
        # 5. Visualisation des métriques cliniques
        plot_clinical_metrics(clinical_metrics, output_dir)
        
        # 6. Analyse des erreurs
        analyze_error_cases(model, test_loader, device, threshold, os.path.join(output_dir, 'error_cases'))
        
        # 7. Si le seuil optimal est différent, recalculer les métriques
        if abs(optimal_threshold - threshold) > 0.01:
            print(f"\nRecalcul des métriques avec le seuil optimal ({optimal_threshold:.2f})...")
            optimal_preds = (all_probs >= optimal_threshold).astype(int)
            optimal_metrics = calculate_clinical_metrics(all_labels, optimal_preds)
            
            print("\nRapport de classification avec seuil optimal:")
            print(classification_report(
                all_labels, optimal_preds,
                target_names=['NORMAL', 'PNEUMONIA']
            ))
    else:
        optimal_threshold = threshold
    
    # Sauvegarder un résumé des métriques
    metrics_summary = {
        "Seuil": threshold,
        "Seuil optimal": optimal_threshold if visualize else threshold,
        "Accuracy": (all_labels == all_preds).mean(),
        "Sensibilité": clinical_metrics['sensitivity'],
        "Spécificité": clinical_metrics['specificity'],
        "VPP": clinical_metrics['ppv'],
        "VPN": clinical_metrics['npv'],
        "Échantillons de test": len(all_labels)
    }
    
    # Sauvegarder en CSV 
    pd.DataFrame([metrics_summary]).to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    print(f"\nRésumé des métriques sauvegardé dans {output_dir}/metrics_summary.csv")
    
    return {
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'metrics': clinical_metrics,
        'optimal_threshold': optimal_threshold
    }

def main():
    """Fonction principale d'exécution"""
    parser = argparse.ArgumentParser(description="Évaluation complète du modèle avec visualisations")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Chemin vers le modèle")
    parser.add_argument("--data_dir", type=str, default="data/raw/chest_xray", help="Répertoire des données")
    parser.add_argument("--output_dir", type=str, default="results", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de classification")
    parser.add_argument("--no_visuals", action="store_true", help="Ne pas générer de visualisations")
    
    args = parser.parse_args()
    
    # Évaluer le modèle
    evaluate(
        threshold=args.threshold,
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        visualize=not args.no_visuals
    )

if __name__ == "__main__":
    main()