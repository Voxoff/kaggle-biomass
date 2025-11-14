import time
def print_result(train_loss, val_loss, epoch_start, epoch, num_epochs, val_r2):
    epoch_time = time.time() - epoch_start
    mins, secs = divmod(epoch_time, 60)
    
    print(f'Epoch {epoch+1}/{num_epochs} - '
          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | '
          f'R²: {val_r2:.4f} | '
          f'Time: {int(mins)}m {int(secs)}s')
