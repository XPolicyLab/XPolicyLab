import importlib


def main():
    app_name = 'visualizers.det_visualizer'
    app = importlib.import_module(app_name)
    app.main()


if __name__ == '__main__':
    main()
