import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from main import main

st.set_page_config(layout="wide")

def streamlit_app():
    st.title("PyTorch Model Training Dashboard")
    
    # Initialize session state for metrics storage
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'train_acc': [],
            'test_acc': [],
            'loss': []
        }
    
    with st.sidebar:
        st.header("Model Architecture Configuration")
        data_path = st.text_input("Dataset Path", "archive")
        
        
        # Conv Layers Configuration
        st.subheader("Convolutional Layers")
        conv_channels = st.multiselect(
            "Channel sizes for conv layers",
            options=[8, 16, 32, 64, 128, 256],
            default=[16, 32, 64]
        )
        
        # FC Layers Configuration
        st.subheader("Fully-Connected Layers")
        fc_channels = st.multiselect(
            "Neurons in FC layers",
            options=[64, 128, 256, 512, 1024],
            default=[128, 256, 512]
        )
        
        # Pooling Type
        pool_type = st.selectbox(
            "Pooling Type",
            options=['max', 'avg', 'adaptive'],
            index=0
        )
        epochs = st.slider("Epochs", 1, 100, 10)
        lr = st.number_input("Learning Rate", 1e-5, 1e-1, 1e-4, format = "%.0e")
        weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 0.01)
        model_type = st.selectbox("Model Architecture", ["ModelV", "Original Model"])
        
        if st.button("Start Training"):
            if not os.path.exists(data_path):
                st.error(f"Directory {data_path} not found!")
                return
            
            st.session_state.training_params = {
                "root": data_path,
                "epoch": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "model_type": model_type
            }
            st.session_state.training_started = True
            st.session_state.metrics = {'train_acc': [], 'test_acc': [], 'loss': []}  # Reset metrics

    # Main display area
    if st.session_state.get("training_started"):
        st.header("Training Progress")
        
        # Create layout columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Metrics table
            st.subheader("Current Metrics")
            metrics_table = st.empty()
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

        with col2:
            # Plot placeholder
            st.subheader("Training Progress")
            plot_placeholder = st.empty()
        
        # Custom callback function
        def update_callback(metrics):
            # Store metrics
            st.session_state.metrics['train_acc'].append(metrics['train_acc'])
            st.session_state.metrics['test_acc'].append(metrics['test_acc'])
            st.session_state.metrics['loss'].append(metrics['loss'])
            
            # Update progress
            progress = metrics['epoch'] / st.session_state.training_params['epoch']
            progress_bar.progress(min(progress, 1.0))
            
            # Update metrics table
            df = pd.DataFrame({
                'Epoch': range(1, metrics['epoch']+1),
                'Train Acc': st.session_state.metrics['train_acc'],
                'Test Acc': st.session_state.metrics['test_acc'],
                'Loss': st.session_state.metrics['loss']
            })
            metrics_table.dataframe(df.tail(5), height=200)
            
            # Update plot
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            # Plot accuracy
            ax1.plot(df['Epoch'], df['Train Acc'], 'b-', label='Train Acc')
            ax1.plot(df['Epoch'], df['Test Acc'], 'g-', label='Test Acc')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)', color='b')
            ax1.tick_params('y', colors='b')
            ax1.set_ylim(0, 100)
            
            # Plot loss on second axis
            ax2 = ax1.twinx()
            ax2.plot(df['Epoch'], df['Loss'], 'r--', label='Loss')
            ax2.set_ylabel('Loss', color='r')
            ax2.tick_params('y', colors='r')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title('Training Progress')
            plot_placeholder.pyplot(fig)
            plt.close()
            
            # Update status
            status_text.text(f"Epoch {metrics['epoch']}/{st.session_state.training_params['epoch']} - "
                           f"Train Acc: {metrics['train_acc']:.2f}%, "
                           f"Test Acc: {metrics['test_acc']:.2f}%, "
                           f"Loss: {metrics['loss']:.4f}")

        # Run training
        try:
            main(
                root = st.session_state.training_params["root"],
                epoch = st.session_state.training_params["epoch"],
                update_callback = update_callback  # Pass our callback
            )
            st.success("Training completed successfully!")
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    streamlit_app()