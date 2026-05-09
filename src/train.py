# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import yaml

# class ModelTrainer:
#     def __init__(self, config_path: str = "config/config.yaml"):
#         try:
#             with open(config_path, 'r') as file:
#                 self.config = yaml.safe_load(file)
#         except FileNotFoundError:
#             print(f"❌ Config file not found at {config_path}")
#             raise
        
#         # Auto-detect number of classes
#         self.num_classes = self._detect_num_classes()
#         print(f"🎯 Auto-detected {self.num_classes} classes")
        
#         self.img_height = self.config['data']['image_height']
#         self.img_width = self.config['data']['image_width']
#         self.batch_size = self.config['data']['batch_size']
#         self.model = None
#         self.history = None
    
#     def _detect_num_classes(self):
#         """Automatically detect number of classes from data directory"""
#         data_dir = self.config['paths']['data_dir']
        
#         if not os.path.exists(data_dir):
#             print(f"⚠️  Data directory {data_dir} not found. Using default 5 classes.")
#             return 5
        
#         # Count subdirectories (each is a class)
#         classes = [d for d in os.listdir(data_dir) 
#                   if os.path.isdir(os.path.join(data_dir, d))]
        
#         num_classes = len(classes)
#         print(f"📁 Found {num_classes} classes: {classes}")
#         return num_classes
    
#     def create_data_generators(self):
#         """Create data generators with auto-detected classes"""
#         data_dir = self.config['paths']['data_dir']
        
#         if not os.path.exists(data_dir):
#             raise FileNotFoundError(f"Data directory {data_dir} not found")
        
#         # Data augmentation for training
#         train_datagen = ImageDataGenerator(
#             rescale=1./255,
#             rotation_range=20,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             horizontal_flip=True,
#             validation_split=0.2
#         )
        
#         # Data generator for validation
#         val_datagen = ImageDataGenerator(
#             rescale=1./255,
#             validation_split=0.2
#         )
        
#         # Create generators
#         train_generator = train_datagen.flow_from_directory(
#             data_dir,
#             target_size=(self.img_height, self.img_width),
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             subset='training',
#             shuffle=True
#         )
        
#         validation_generator = val_datagen.flow_from_directory(
#             data_dir,
#             target_size=(self.img_height, self.img_width),
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             subset='validation',
#             shuffle=False
#         )
        
#         # Update num_classes based on actual data
#         # update the train model to reflect the actual number of classes
#         actual_classes = len(train_generator.class_indices)
#         if actual_classes != self.num_classes:
#             print(f"🔄 Updating num_classes from {self.num_classes} to {actual_classes}")
#             self.num_classes = actual_classes
        
#         print(f"✅ Training samples: {train_generator.samples}")
#         print(f"✅ Validation samples: {validation_generator.samples}")
#         print(f"✅ Classes: {list(train_generator.class_indices.keys())}")
        
#         return train_generator, validation_generator
    
#     def create_model(self, model_type='simple'):
#         """Create model with correct number of output classes"""
        
#         if model_type == 'simple':
#             model = tf.keras.Sequential([
#                 tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
#                                       input_shape=(self.img_height, self.img_width, 3)),
#                 tf.keras.layers.MaxPooling2D(2, 2),
                
#                 tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#                 tf.keras.layers.MaxPooling2D(2, 2),
                
#                 tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#                 tf.keras.layers.MaxPooling2D(2, 2),
                
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dense(512, activation='relu'),
#                 tf.keras.layers.Dropout(0.5),
#                 tf.keras.layers.Dense(self.num_classes, activation='softmax')
#             ])
#         elif model_type == 'resnet':
#             base_model = tf.keras.applications.ResNet50(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(self.img_height, self.img_width, 3)
#             )
#             base_model.trainable = False
            
#             model = tf.keras.Sequential([
#                 base_model,
#                 tf.keras.layers.GlobalAveragePooling2D(),
#                 tf.keras.layers.Dense(256, activation='relu'),
#                 tf.keras.layers.Dropout(0.5),
#                 tf.keras.layers.Dense(self.num_classes, activation='softmax')
#             ])
        
#         model.compile(
#             optimizer='adam',
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )
        
#         print(f"✅ Created {model_type} model with {self.num_classes} output classes")
#         return model
    
#     def setup_callbacks(self):
#         """Setup training callbacks"""
#         os.makedirs(self.config['paths']['model_dir'], exist_ok=True)
#         os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=5,
#                 restore_best_weights=True,
#                 verbose=1
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.2,
#                 patience=3,
#                 min_lr=1e-7,
#                 verbose=1
#             ),
#             ModelCheckpoint(
#                 filepath=os.path.join(self.config['paths']['model_dir'], 'best_model.h5'),
#                 monitor='val_accuracy',
#                 save_best_only=True,
#                 verbose=1
#             )
#         ]
        
#         return callbacks
    
#     def train_model(self, model_type='simple', fine_tune=False):
#         """Train the model"""
#         print(f"🚀 Starting {model_type} model training...")
#         print(f"📊 Number of classes: {self.num_classes}")
        
#         try:
#             # Create data generators
#             train_generator, validation_generator = self.create_data_generators()
            
#             # Create model
#             self.model = self.create_model(model_type)
            
#             # Setup callbacks
#             callbacks = self.setup_callbacks()
            
#             print("\n📈 Starting training...")
#             self.history = self.model.fit(
#                 train_generator,
#                 epochs=self.config['model']['epochs'],
#                 validation_data=validation_generator,
#                 callbacks=callbacks,
#                 verbose=1
#             )
            
#             # Save final model
#             model_path = os.path.join(self.config['paths']['model_dir'], 'final_model.h5')
#             self.model.save(model_path)
#             print(f"✅ Model saved to {model_path}")
            
#             return self.history
            
#         except Exception as e:
#             print(f"❌ Training error: {e}")
#             print("💡 Creating demo with synthetic data...")
#             return self._train_with_demo_data()
    
#     def _train_with_demo_data(self):
#         """Train with synthetic data for demonstration"""
#         print("🎭 Training with synthetic data for demonstration...")
        
#         # Create synthetic data matching the detected number of classes
#         num_samples = 200
#         X_train = np.random.random((num_samples, self.img_height, self.img_width, 3))
#         y_train = tf.keras.utils.to_categorical(
#             np.random.randint(0, self.num_classes, num_samples), 
#             num_classes=self.num_classes
#         )
        
#         # Create model
#         self.model = self.create_model('simple')
        
#         # Train on synthetic data
#         self.history = self.model.fit(
#             X_train, y_train,
#             epochs=5,
#             validation_split=0.2,
#             verbose=1,
#             batch_size=32
#         )
        
#         print("✅ Demo training completed!")
#         return self.history
    
#     def plot_training_history(self):
#         """Plot training history"""
#         if self.history is None:
#             print("❌ No training history available.")
#             return
        
#         plt.figure(figsize=(12, 4))
        
#         plt.subplot(1, 2, 1)
#         plt.plot(self.history.history['accuracy'], label='Training Accuracy')
#         plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
#         plt.title('Model Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()
#         plt.grid(True)
        
#         plt.subplot(1, 2, 2)
#         plt.plot(self.history.history['loss'], label='Training Loss')
#         plt.plot(self.history.history['val_loss'], label='Validation Loss')
#         plt.title('Model Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True)
        
#         plt.tight_layout()
#         plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
#         plt.show()

# def main():
#     """Main function"""
#     print("🔧 Solar Panel Fault Detection - Training")
#     trainer = ModelTrainer()
#     trainer.train_model('simple')
#     trainer.plot_training_history()

# if __name__ == "__main__":
#     main()

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datetime import datetime

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"❌ Config file not found at {config_path}")
            print("📝 Using default configuration...")
            self.config = self._get_default_config()
        
        # Auto-detect number of classes
        self.num_classes = self._detect_num_classes()
        print(f"🎯 Auto-detected {self.num_classes} classes")
        
        self.img_height = self.config['data']['image_height']
        self.img_width = self.config['data']['image_width']
        self.batch_size = self.config['data']['batch_size']
        self.model = None
        self.history = None
        self.class_names = []
    
    def _get_default_config(self):
        """Return default configuration"""
        return {
            'data': {
                'image_height': 224,
                'image_width': 224,
                'batch_size': 32
            },
            'model': {
                'epochs': 50,
                'learning_rate': 0.001
            },
            'paths': {
                'data_dir': 'data/raw',
                'model_dir': 'models/trained_models',
                'logs_dir': 'logs'
            }
        }
    
    def _detect_num_classes(self):
        """Automatically detect number of classes from data directory"""
        data_dir = self.config['paths']['data_dir']
        
        if not os.path.exists(data_dir):
            print(f"⚠️ Data directory {data_dir} not found.")
            print(f"📁 Creating directory structure...")
            os.makedirs(data_dir, exist_ok=True)
            print(f"💡 Please organize your images as:")
            print(f"   {data_dir}/class_1/")
            print(f"   {data_dir}/class_2/")
            print(f"   etc.")
            return 2  # Default for binary classification
        
        # Count subdirectories (each is a class)
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        
        if len(classes) == 0:
            print(f"⚠️ No class subdirectories found in {data_dir}")
            print(f"💡 Create folders like: {data_dir}/Dust/, {data_dir}/Healthy/, etc.")
            return 2
        
        num_classes = len(classes)
        print(f"📁 Found {num_classes} classes: {classes}")
        return num_classes
    
    def create_data_generators(self):
        """Create data generators with auto-detected classes"""
        data_dir = self.config['paths']['data_dir']
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        
        # Check if there are any images
        has_images = False
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    has_images = True
                    break
            if has_images:
                break
        
        if not has_images:
            raise ValueError(f"No images found in {data_dir}. Please add images to class subdirectories.")
        
        # Enhanced data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Update num_classes and store class names
        self.class_names = list(train_generator.class_indices.keys())
        actual_classes = len(self.class_names)
        
        if actual_classes != self.num_classes:
            print(f"🔄 Updating num_classes from {self.num_classes} to {actual_classes}")
            self.num_classes = actual_classes
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   ✅ Training samples: {train_generator.samples}")
        print(f"   ✅ Validation samples: {validation_generator.samples}")
        print(f"   ✅ Classes: {self.class_names}")
        print(f"   ✅ Class indices: {train_generator.class_indices}")
        
        # Check for class imbalance
        class_counts = {}
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = count
                print(f"   📸 {class_name}: {count} images")
        
        min_count = min(class_counts.values()) if class_counts else 0
        max_count = max(class_counts.values()) if class_counts else 0
        if max_count > 2 * min_count:
            print(f"   ⚠️ Warning: Class imbalance detected! (min={min_count}, max={max_count})")
            print(f"   💡 Consider using class weights or collecting more data for minority classes")
        
        return train_generator, validation_generator
    
    def create_model(self, model_type='simple'):
        """Create model with correct number of output classes"""
        
        if model_type == 'simple':
            model = models.Sequential([
                # First convolutional block
                layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                             input_shape=(self.img_height, self.img_width, 3)),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D(2, 2),
                layers.Dropout(0.25),
                
                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D(2, 2),
                layers.Dropout(0.25),
                
                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D(2, 2),
                layers.Dropout(0.25),
                
                # Fourth convolutional block
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2, 2),
                layers.Dropout(0.25),
                
                # Classifier
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            optimizer = Adam(learning_rate=self.config['model'].get('learning_rate', 0.001))
            
        elif model_type == 'resnet':
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            base_model.trainable = False
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            optimizer = Adam(learning_rate=self.config['model'].get('learning_rate', 0.0001))
        
        else:  # efficientnet
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            base_model.trainable = False
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            optimizer = Adam(learning_rate=self.config['model'].get('learning_rate', 0.0001))
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        print(f"✅ Created {model_type} model with {self.num_classes} output classes")
        print(f"📊 Model summary:")
        model.summary()
        
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        os.makedirs(self.config['paths']['model_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                cooldown=2
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.config['paths']['model_dir'], f'best_model_{timestamp}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            CSVLogger(
                filename=os.path.join(self.config['paths']['logs_dir'], f'training_log_{timestamp}.csv'),
                separator=',',
                append=False
            )
        ]
        
        return callbacks
    
    def train_model(self, model_type='simple', fine_tune=False):
        """Train the model"""
        print(f"\n🚀 Starting {model_type} model training...")
        print(f"📊 Number of classes: {self.num_classes}")
        print("=" * 60)
        
        try:
            # Create data generators
            train_generator, validation_generator = self.create_data_generators()
            
            # Check minimum samples requirement
            if train_generator.samples < self.batch_size:
                print(f"❌ Not enough training samples ({train_generator.samples} < {self.batch_size})")
                print("💡 Please add more images or reduce batch size")
                return None
            
            # Create model
            self.model = self.create_model(model_type)
            
            # Compute class weights for imbalance
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_generator.classes),
                y=train_generator.classes
            )
            class_weight_dict = dict(enumerate(class_weights))
            print(f"⚖️ Class weights: {class_weight_dict}")
            
            # Setup callbacks
            callbacks = self.setup_callbacks()
            
            print("\n📈 Starting training...")
            print(f"   Epochs: {self.config['model']['epochs']}")
            print(f"   Batch size: {self.batch_size}")
            print(f"   Training steps: {train_generator.samples // self.batch_size}")
            print(f"   Validation steps: {validation_generator.samples // self.batch_size}")
            
            self.history = self.model.fit(
                train_generator,
                epochs=self.config['model']['epochs'],
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1,
                class_weight=class_weight_dict
            )
            
            # Save final model and class names
            os.makedirs('models/trained_models', exist_ok=True)
            
            # Save in multiple formats for compatibility
            model_path = os.path.join(self.config['paths']['model_dir'], 'solar_fault_model.h5')
            self.model.save(model_path)
            print(f"✅ Model saved to {model_path}")
            
            # Also save as final_model.h5 for compatibility with predict.py
            final_model_path = os.path.join(self.config['paths']['model_dir'], 'final_model.h5')
            self.model.save(final_model_path)
            print(f"✅ Model saved to {final_model_path}")
            
            # Save class names
            class_names_path = os.path.join(self.config['paths']['model_dir'], 'class_names.npy')
            np.save(class_names_path, np.array(self.class_names))
            print(f"✅ Class names saved to {class_names_path}")
            
            # Evaluate final model
            print("\n📊 Final Evaluation:")
            test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
                validation_generator, verbose=0
            )
            print(f"   Validation Accuracy: {test_accuracy:.2%}")
            print(f"   Validation Precision: {test_precision:.2%}")
            print(f"   Validation Recall: {test_recall:.2%}")
            print(f"   Validation Loss: {test_loss:.4f}")
            
            if test_accuracy < 0.6:
                print("\n⚠️ WARNING: Model accuracy is below 60%")
                print("   Possible reasons:")
                print("   1. Not enough training data")
                print("   2. Poor image quality")
                print("   3. Classes are too similar")
                print("   4. Try using --model_type resnet or efficientnet")
            
            return self.history
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 To fix this issue:")
            print("   1. Ensure you have images in data/raw/class_name/ folders")
            print("   2. Each class folder should contain at least 10 images")
            print("   3. Supported formats: .jpg, .jpeg, .png, .bmp")
            return None
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision', linewidth=2)
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall', linewidth=2)
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✅ Training history plot saved as 'training_history.png'")
        plt.show()
    
    def predict_single_image(self, image_path):
        """Test prediction on a single image"""
        if self.model is None:
            print("❌ No trained model. Please train first.")
            return None
        
        from tensorflow.keras.preprocessing import image
        img = image.load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        print(f"\n🧪 Test Prediction:")
        print(f"   Image: {image_path}")
        print(f"   Predicted: {self.class_names[predicted_class_idx]}")
        print(f"   Confidence: {confidence:.2%}")
        
        return predicted_class_idx, confidence


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Solar Panel Fault Detection - Training')
    parser.add_argument('--model_type', type=str, default='simple', 
                       choices=['simple', 'resnet', 'efficientnet'],
                       help='Model architecture to use')
    parser.add_argument('--test_image', type=str, default=None,
                       help='Optional image to test after training')
    
    args = parser.parse_args()
    
    print("🔧 Solar Panel Fault Detection - Training System")
    print("=" * 60)
    
    trainer = ModelTrainer()
    history = trainer.train_model(model_type=args.model_type)
    
    if history is not None:
        trainer.plot_training_history()
        
        if args.test_image and os.path.exists(args.test_image):
            trainer.predict_single_image(args.test_image)
        
        print("\n✅ Training completed successfully!")
        print("💡 To make predictions, run:")
        print("   python main.py --mode predict --image path/to/image.jpg")
    else:
        print("\n❌ Training failed!")
        print("💡 Please ensure you have proper data in data/raw/ folder")

if __name__ == "__main__":
    main()