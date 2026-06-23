import os
import time
import pyautogui

# visuals folder check karna jahan image save hogi
os.makedirs('visuals', exist_ok=True)

print("🔄 5 seconds ka time hai... Jaldi se apna Streamlit Dashboard screen par open karlein!")
# 5 seconds ka delay takay aap dashboard wali window samne la sakein
time.sleep(5)

# Screenshot lena
screenshot = pyautogui.screenshot()

# Sahi folder aur naam se save karna
output_path = os.path.join('visuals', 'dashboard_home.png')
screenshot.save(output_path)

print(f"🎯 Done! Deployed application view/screenshot direct '{output_path}' mein save ho gaya hai.")