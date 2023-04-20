# http://www.l2xin.com/index.php/python/python-print-colored/

class Colored(object):
    from colorama import  init, Fore, Back, Style
    init(autoreset=True)

    #  前景色:红色  背景色:默认
    @staticmethod
    def red(s):
        return Fore.RED + s + Fore.RESET

    #  前景色:绿色  背景色:默认
    @staticmethod
    def green(s):
        return Fore.GREEN + s + Fore.RESET

    #  前景色:黄色  背景色:默认
    @staticmethod
    def yellow(s):
        return Fore.YELLOW + s + Fore.RESET

    #  前景色:蓝色  背景色:默认
    @staticmethod
    def blue(s):
        return Fore.BLUE + s + Fore.RESET

    #  前景色:洋红色  背景色:默认
    @staticmethod
    def magenta(s):
        return Fore.MAGENTA + s + Fore.RESET

    #  前景色:青色  背景色:默认
    @staticmethod
    def cyan(s):
        return Fore.CYAN + s + Fore.RESET

    #  前景色:白色  背景色:默认
    @staticmethod
    def white(s):
        return Fore.WHITE + s + Fore.RESET

    #  前景色:黑色  背景色:默认
    @staticmethod
    def black(s):
        return Fore.BLACK

    #  前景色:白色  背景色:绿色
    @staticmethod
    def white_green(s):
        return Fore.WHITE + Back.GREEN + s + Fore.RESET + Back.RESET


