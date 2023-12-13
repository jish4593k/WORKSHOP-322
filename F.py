import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.img = None

    def load_image(self, image_path):
        self.img = cv2.imread(image_path)
        return self.img

    def resize(self, scale_percent):
        width = int(self.img.shape[1] * int(scale_percent) / 100)
        height = int(self.img.shape[0] * int(scale_percent) / 100)
        dim = (width, height)
        img_resized = cv2.resize(self.img, dim, interpolation=cv2.INTER_NEAREST)
        return img_resized

    def blur(self, blur_choice, k1=None, k2=None):
        if blur_choice == "1":
            img_blurred = cv2.blur(self.img, (int(k1), int(k2)))
        elif blur_choice == "2":
            img_blurred = cv2.medianBlur(self.img, int(k1))
        elif blur_choice == "3":
            img_blurred = cv2.GaussianBlur(self.img, (5, 5), 0)
        else:
            print("Not found")
            return self.img

        return img_blurred

    def convert_color(self, color_choice):
        if color_choice == "1":
            img_colored = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif color_choice == "2":
            img_colored = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        elif color_choice == "3":
            img_colored = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        else:
            print("Not found")
            return self.img

        return img_colored

    def find_contours(self, contours_choice, t_min=None, t_max=None):
        if contours_choice == "1":
            img_contoured = cv2.Canny(self.img, int(t_min), int(t_max))
        elif contours_choice == "2":
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            ret, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_contoured = np.zeros_like(self.img)
            img_contoured = cv2.drawContours(img_contoured, contours, -1, (0, 255, 0), 3)
        else:
            print("Not found")
            return self.img

        return img_contoured


class ImageClassifier(nn.Module):
    def __init__(self, model_path, class_names_path):
        super(ImageClassifier, self).__init__()
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()
        self.class_names = self.load_class_names(class_names_path)

    def load_class_names(self, file_path):
        with open(file_path, "r") as file:
            return file.read().split(",")

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)

    def predict_image(self, image_path):
        data = self.preprocess_image(image_path)
        with torch.no_grad():
            prediction = self.model(data)
        index = torch.argmax(prediction).item()
        class_name = self.class_names[index]
        confidence_score = torch.softmax(prediction, dim=1)[0][index].item()
        return class_name, confidence_score


class GUI(tk.Tk):
    def __init__(self, image_processor, image_classifier):
        super().__init__()

        self.image_processor = image_processor
        self.image_classifier = image_classifier

        self.title("Image Processor and Classifier")
        self.geometry("800x600")

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        self.operations_label = tk.Label(self, text="Choose an operation:")
        self.operations_label.pack()

        self.operations_var = tk.StringVar()
        self.operations_var.set("1")  # Default to the Resize operation
        self.operations_menu = tk.OptionMenu(self, self.operations_var, "1", "2", "3", "4")
        self.operations_menu.pack(pady=10)

        self.execute_button = tk.Button(self, text="Execute Operation", command=self.execute_operation)
        self.execute_button.pack(pady=10)

        self.quit_button = tk.Button(self, text="Quit", command=self.destroy)
        self.quit_button.pack(pady=10)

    def execute_operation(self):
        operation_choice = self.operations_var.get()

        if operation_choice == "1":
            self.resize_operation()
        elif operation_choice == "2":
            self.blur_operation()
        elif operation_choice == "3":
            self.color_operation()
        elif operation_choice == "4":
            self.contours_operation()

    def resize_operation(self):
        scale_percent = simpledialog.askstring("Resize", "Enter Scale Percent:")
        img_resized = self.image_processor.resize(scale_percent)
        self.display_image(img_resized)

    def blur_operation(self):
        blur_choice = simpledialog.askstring("Blur", "1. Blur\n2. Median Blur\n3. Gaussian Blur\nEnter your choice:")
        if blur_choice in ["1", "2", "3"]:
            k1 = simpledialog.askstring("Blur", "Enter K size 1:")
            k2 = simpledialog.askstring("Blur", "Enter K size 2:")
            img_blurred = self.image_processor.blur(blur_choice, k1, k2)
            self.display_image(img_blurred)
        else:
            messagebox.showinfo("Invalid Choice", "Invalid blur choice. Choose 1, 2, or 3.")

    def color_operation(self):
        color_choice = simpledialog.askstring("Color", "1. Gray\n2. LAB\n3. HSV\nEnter your choice:")
        if color_choice in ["1", "2", "3"]:
            img_colored = self.image_processor.convert_color(color_choice)
            self.display_image(img_colored)
        else:
            messagebox.showinfo("Invalid Choice", "Invalid color choice. Choose 1, 2, or 3.")

    def contours_operation(self):
        contours_choice = simpledialog.askstring("Contours", "1. Canny\n2. Find Contours\nEnter your choice:")
        if contours_choice in ["1", "2"]:
            if contours_choice == "1":
                t_min = simpledialog.askstring("Contours", "Enter threshold min:")
                t_max = simpledialog.askstring("Contours", "Enter threshold max:")
            else:
                t_min, t_max = None, None

            img_contoured = self.image_processor.find_contours(contours_choice, t_min, t_max)
            self.display_image(img_contoured)
        else:
            messagebox.showinfo("Invalid Choice", "Invalid contours choice. Choose 1 or 2.")

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img


def main():
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    image_processor = ImageProcessor()
    image_classifier = ImageClassifier(model_path='your_model.pth', class_names_path='classnames.txt')

    gui = GUI(image_processor, image_classifier)
    gui.image_processor.load_image(image_path)  # Load the initial image
    gui.display_image(gui.image_processor.img)
    gui.mainloop()

if __name__ == "__main__":
    main()
