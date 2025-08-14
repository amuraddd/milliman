def notebook_line_magic():
    """
    Avoid having to restart kernel when working with python scripts
    """
    from IPython import get_ipython
    ip = get_ipython()
    ip.run_line_magic("reload_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    print("Line Magic Set")
    