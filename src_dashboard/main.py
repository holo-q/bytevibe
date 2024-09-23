import dearpygui.dearpygui as dpg
from training_window import TrainingWindow
from checkpoint_manager import CheckpointManager
from src_dashboard.run_manager import singleton as run_manager

# noinspection PyArgumentList
class MainApplication:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
        self.training_windows = []

    def setup(self):
        dpg.create_context()

        with dpg.window(label="Model Training Dashboard", tag="primary_window"):
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Load Run", callback=self.load_run)
                    dpg.add_menu_item(label="Exit", callback=dpg.stop_dearpygui)

            with dpg.group():
                dpg.add_text("Active Runs:")
                self.active_runs_list = dpg.add_listbox([], callback=self.on_select_run, num_items=5)

            self.txt_newrun_name = dpg.add_input_text(label="Run Name")
            self.txt_newrun_model = dpg.add_input_text(label="Run Name")
            dpg.add_button(label="Create", callback=self.on_confirm_create_run)

            with dpg.group():
                dpg.add_text("System Info:")
                self.gpu_info = dpg.add_text("GPU: Checking...")
                self.memory_info = dpg.add_text("Memory: Checking...")

        # Create popup window
        # with dpg.window(label="Create New Run", show=False, modal=True, tag="create_run_popup"):

        dpg.create_viewport(title="Enpetri", width=1200, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def on_confirm_create_run(self):
        run_name = dpg.get_value(self.txt_newrun_name)
        model_name = dpg.get_value(self.txt_newrun_model)

        run_manager.create_run(run_name, model_name)
        self.refresh_active_runs_list()
        dpg.configure_item('create_run_popup', show=False)

        new_window = TrainingWindow()
        new_window.set_run(run_manager.get_current_run())
        self.training_windows.append(new_window)
        new_window.setup()

    def on_select_run(self, sender, app_data):
        # Implementation for selecting a run
        pass

    def load_run(self):
        # Implementation for loading a run
        pass

    def refresh_active_runs_list(self):
        active_runs = run_manager.get_active_runs()
        dpg.configure_item(self.active_runs_list, items=active_runs)


    # def on_create_training_window(self):

    def update_system_info(self):
        import psutil
        import torch

        gpu_info = f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}"
        memory_info = f"Memory: {psutil.virtual_memory().percent}% used"

        dpg.set_value(self.gpu_info, gpu_info)
        dpg.set_value(self.memory_info, memory_info)

    def run(self):
        while dpg.is_dearpygui_running():
            for window in self.training_windows:
                window.update()
            self.update_system_info()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

if __name__ == "__main__":
    app = MainApplication()
    app.setup()
    app.run()