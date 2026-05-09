# #!/usr/bin/env python3
# """
# SOLAR PANEL FAULT DETECTION - COMPLETE WORKING VERSION
# """
# import os
# import sys
# import argparse
# import glob
# sys.path.append('src')

# def main():
#     parser = argparse.ArgumentParser(description='Solar Panel Fault Detection')
#     parser.add_argument('--mode', type=str, required=True,
#                        choices=['train', 'predict', 'demo', 'setup', 'list-images'],
#                        help='Mode: train, predict, demo, setup, or list-images')
#     parser.add_argument('--image', type=str, help='Path to image file for prediction')
    
#     args = parser.parse_args()
    
#     print("🔧 SOLAR PANEL FAULT DETECTION SYSTEM")
#     print("=" * 60)
    
#     try:
#         if args.mode == 'train':
#             train_model()
            
#         elif args.mode == 'predict':
#             if not args.image:
#                 print("❌ ERROR: Please provide an image file with --image")
#                 print("💡 Example: python main.py --mode predict --image my_image.jpg")
#                 print("💡 Or use: python main.py --mode list-images to see available images")
#                 return
#             predict_image(args.image)
            
#         elif args.mode == 'demo':
#             run_demo()
            
#         elif args.mode == 'setup':
#             check_project_setup()
            
#         elif args.mode == 'list-images':
#             list_available_images()
            
#     except Exception as e:
#         print(f"❌ ERROR: {e}")
#         import traceback
#         traceback.print_exc()

# def train_model():
#     """Train the solar panel fault detection model"""
#     from train import ModelTrainer
#     print("🚀 Starting training process...")
#     trainer = ModelTrainer()
#     trainer.train_model()
    
#     if trainer.history is not None:
#         trainer.plot_training_history()
#         print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        
#         # Show where model was saved
#         model_path = 'models/trained_models/solar_fault_model.h5'
#         if os.path.exists(model_path):
#             print(f"💾 Model saved: {model_path}")
#     else:
#         print("⚠️ Training finished but no real data was used.")

# def predict_image(image_path):
#     """Predict fault type for a single image"""
#     try:
#         from tensorflow.keras.preprocessing import image
#         import numpy as np
#         import tensorflow as tf
        
#         # Check if file exists
#         if not os.path.exists(image_path):
#             print(f"❌ ERROR: File not found: {image_path}")
#             print("💡 Available images:")
#             list_available_images()
#             return
        
#         # Check if model exists
#         model_path = 'models/trained_models/solar_fault_model.h5'
#         if not os.path.exists(model_path):
#             print("❌ No trained model found.")
#             print("💡 Please train the model first:")
#             print("   python main.py --mode train")
#             return
        
#         print(f"📷 Loading image: {image_path}")
#         img = image.load_img(image_path, target_size=(224, 224))
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
        
#         print("🤖 Loading trained model...")
#         model = tf.keras.models.load_model(model_path)
        
#         print("🔮 Making prediction...")
#         predictions = model.predict(img_array, verbose=0)
        
#         # Define class names
#         class_names = ['Dust', 'Snow', 'Bird Droppings', 'Crack', 'Healthy']
        
#         # Get results
#         predicted_class_idx = np.argmax(predictions[0])
#         predicted_class = class_names[predicted_class_idx]
#         confidence = np.max(predictions[0])
        
#         print("\n" + "=" * 50)
#         print("🎯 PREDICTION RESULTS")
#         print("=" * 50)
#         print(f"✅ Fault Type: {predicted_class}")
#         print(f"📊 Confidence: {confidence:.2%}")
#         print("\n📈 All probabilities:")
#         for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
#             print(f"   {class_name}: {prob:.2%}")
#         print("=" * 50)
        
#     except Exception as e:
#         print(f"❌ Prediction error: {e}")

# def run_demo():
#     """Run demo with synthetic data"""
#     from train import ModelTrainer
#     print("🎭 Running demo with synthetic data...")
#     trainer = ModelTrainer()
#     trainer._train_with_demo_data()
#     trainer.plot_training_history()

# def list_available_images():
#     """List all available images in the project"""
#     print("📸 AVAILABLE IMAGES:")
#     print("=" * 40)
    
#     # Check data directory
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
#     found_images = False
    
#     for root, dirs, files in os.walk('data'):
#         for file in files:
#             if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
#                 print(f"📁 {os.path.join(root, file)}")
#                 found_images = True
    
#     if not found_images:
#         print("❌ No images found in data directory")
#         print("💡 You can:")
#         print("   1. Add images to data/raw/ folders")
#         print("   2. Use any image file: python main.py --mode predict --image path/to/your/image.jpg")
#         print("   3. Run demo: python main.py --mode demo")

# def check_project_setup():
#     """Check project structure"""
#     print("🔍 PROJECT SETUP CHECK")
#     print("=" * 50)
    
#     required = {
#         'Files': ['main.py', 'train_model.py', 'requirements.txt', 'config/config.yaml'],
#         'Directories': ['src/', 'data/raw/', 'models/']
#     }
    
#     all_ok = True
    
#     for category, items in required.items():
#         print(f"\n📁 {category}:")
#         for item in items:
#             if os.path.exists(item):
#                 print(f"   ✅ {item}")
#             else:
#                 print(f"   ❌ {item}")
#                 all_ok = False
    
#     print("\n" + "=" * 50)
#     if all_ok:
#         print("🎉 PROJECT SETUP IS COMPLETE!")
#     else:
#         print("⚠️ Some files/directories are missing")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
SOLAR PANEL FAULT DETECTION - COMPLETE WORKING VERSION
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('src')

def main():
    parser = argparse.ArgumentParser(description='Solar Panel Fault Detection')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'demo', 'setup', 'list-images', 'test'],
                       help='Mode: train, predict, demo, setup, list-images, or test')
    parser.add_argument('--image', type=str, help='Path to image file for prediction')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'resnet', 'efficientnet'],
                       help='Model architecture (for training)')
    
    args = parser.parse_args()
    
    print("🔧 SOLAR PANEL FAULT DETECTION SYSTEM")
    print("=" * 60)
    
    try:
        if args.mode == 'train':
            train_model(model_type=args.model_type)
            
        elif args.mode == 'predict':
            if not args.image:
                print("❌ ERROR: Please provide an image file with --image")
                print("💡 Example: python main.py --mode predict --image my_image.jpg")
                return
            predict_image(args.image)
            
        elif args.mode == 'demo':
            run_demo()
            
        elif args.mode == 'setup':
            check_project_setup()
            
        elif args.mode == 'list-images':
            list_available_images()
            
        elif args.mode == 'test':
            test_model()
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def train_model(model_type='simple'):
    """Train the solar panel fault detection model"""
    from train import ModelTrainer
    
    print(f"🚀 Starting training with {model_type} model...")
    trainer = ModelTrainer()
    history = trainer.train_model(model_type=model_type)
    
    if history is not None:
        trainer.plot_training_history()
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        
        # Show where model was saved
        model_paths = [
            'models/trained_models/solar_fault_model.h5',
            'models/trained_models/final_model.h5'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                print(f"💾 Model saved: {model_path} ({file_size:.2f} MB)")
        
        # Show class names
        class_names_path = 'models/trained_models/class_names.npy'
        if os.path.exists(class_names_path):
            class_names = np.load(class_names_path, allow_pickle=True)
            print(f"📋 Classes: {list(class_names)}")
    else:
        print("⚠️ Training failed. Please check your data directory.")

def predict_image(image_path):
    """Predict fault type for a single image"""
    try:
        from tensorflow.keras.preprocessing import image
        import numpy as np
        import tensorflow as tf
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ ERROR: File not found: {image_path}")
            print("💡 Available images:")
            list_available_images()
            return
        
        # Check if model exists
        model_paths = [
            'models/trained_models/solar_fault_model.h5',
            'models/trained_models/final_model.h5',
            'models/trained_models/best_model.h5'
        ]
        
        model = None
        selected_path = None
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path)
                    selected_path = path
                    break
                except Exception as e:
                    print(f"⚠️ Could not load {path}: {e}")
        
        if model is None:
            print("❌ No valid trained model found.")
            print("💡 Please train the model first:")
            print("   python main.py --mode train")
            return
        
        print(f"✅ Loaded model from: {selected_path}")
        
        # Load class names
        class_names_path = 'models/trained_models/class_names.npy'
        if os.path.exists(class_names_path):
            class_names = np.load(class_names_path, allow_pickle=True)
            print(f"📋 Loaded classes: {list(class_names)}")
        else:
            # Default classes based on model output size
            output_size = model.output_shape[-1]
            default_classes = ['Dust', 'Snow', 'Bird Droppings', 'Crack', 'Healthy']
            class_names = default_classes[:output_size]
            print(f"📋 Using default classes: {list(class_names)}")
        
        print(f"📷 Loading image: {image_path}")
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print("🔮 Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = np.max(predictions[0])
        
        print("\n" + "=" * 60)
        print("🎯 PREDICTION RESULTS")
        print("=" * 60)
        print(f"📷 Image: {os.path.basename(image_path)}")
        print(f"✅ Predicted Fault Type: {predicted_class}")
        print(f"📊 Confidence: {confidence:.2%}")
        
        # Confidence interpretation
        if confidence > 0.8:
            print("   👍 High confidence - Reliable prediction")
        elif confidence > 0.6:
            print("   👌 Moderate confidence - Probably correct")
        elif confidence > 0.4:
            print("   🤔 Low confidence - Consider verification")
        else:
            print("   ⚠️ Very low confidence - Model uncertain")
        
        print("\n📈 All class probabilities:")
        print("-" * 40)
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {class_name:20} {bar} {prob:.2%}")
        print("=" * 60)
        
        # Check if confidence is too low
        if confidence < 0.6:
            print("\n⚠️ WARNING: Confidence is low (<60%)")
            print("   Possible reasons:")
            print("   1. Image is different from training data")
            print("   2. Not enough training data for this class")
            print("   3. Image quality is poor")
            print("   4. Try retraining with more data")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()

def test_model():
    """Test model on sample images if available"""
    print("🧪 Testing Model Performance")
    print("=" * 60)
    
    # Find test images
    test_images = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 5:  # Limit to 5 test images
                    break
        if len(test_images) >= 5:
            break
    
    if not test_images:
        print("❌ No test images found in data directory")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    for img_path in test_images:
        print("\n" + "-" * 40)
        predict_image(img_path)
        print("-" * 40)

def run_demo():
    """Run demo with synthetic data"""
    from train import ModelTrainer
    print("🎭 Running demo with synthetic data...")
    print("⚠️ Note: Demo uses random data for demonstration only")
    print("   Real predictions will have low confidence (~20% for 5 classes)")
    
    trainer = ModelTrainer()
    history = trainer._train_with_demo_data()
    
    if history:
        trainer.plot_training_history()
        print("\n⚠️ This demo model is trained on RANDOM data")
        print("   It will NOT make accurate predictions on real images")
        print("   To get real predictions, collect actual images and run:")
        print("   python main.py --mode train")
    else:
        print("❌ Demo failed")

def list_available_images():
    """List all available images in the project"""
    print("📸 AVAILABLE IMAGES:")
    print("=" * 60)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    found_images = False
    image_count = 0
    
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.lower().endswith(image_extensions):
                rel_path = os.path.relpath(os.path.join(root, file), start='.')
                file_size = os.path.getsize(os.path.join(root, file)) / 1024  # KB
                print(f"   📁 {rel_path} ({file_size:.1f} KB)")
                found_images = True
                image_count += 1
    
    if not found_images:
        print("❌ No images found in data directory")
        print("\n💡 To add images:")
        print("   1. Create class folders: mkdir -p data/raw/Dust data/raw/Healthy")
        print("   2. Add images to respective folders")
        print("   3. Or use: python main.py --mode predict --image your_image.jpg")
    else:
        print(f"\n✅ Found {image_count} images total")
        print("\n💡 To make predictions:")
        print("   python main.py --mode predict --image path/to/your/image.jpg")

def check_project_setup():
    """Check project structure"""
    print("🔍 PROJECT SETUP CHECK")
    print("=" * 60)
    
    required = {
        'Files': ['main.py', 'train.py', 'requirements.txt'],
        'Directories': ['src/', 'data/raw/', 'models/trained_models/', 'logs/']
    }
    
    all_ok = True
    
    for category, items in required.items():
        print(f"\n📁 {category}:")
        for item in items:
            if os.path.exists(item):
                print(f"   ✅ {item}")
            else:
                print(f"   ❌ {item} (will be created automatically)")
                if category == 'Directories':
                    os.makedirs(item, exist_ok=True)
                    print(f"   📁 Created: {item}")
    
    # Check for config file
    if not os.path.exists('config/config.yaml'):
        print("\n⚠️ config/config.yaml not found")
        print("   Creating default config...")
        os.makedirs('config', exist_ok=True)
        with open('config/config.yaml', 'w') as f:
            f.write("""
data:
  image_height: 224
  image_width: 224
  batch_size: 32

model:
  epochs: 50
  learning_rate: 0.001

paths:
  data_dir: data/raw
  model_dir: models/trained_models
  logs_dir: logs
""")
        print("   ✅ Created default config/config.yaml")
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT SETUP CHECK COMPLETE!")
    print("\n💡 Next steps:")
    print("   1. Add images to data/raw/class_name/ folders")
    print("   2. Train model: python main.py --mode train")
    print("   3. Predict: python main.py --mode predict --image your_image.jpg")

if __name__ == "__main__":
    main()