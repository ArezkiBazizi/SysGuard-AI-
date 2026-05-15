"""
Préprocesseur SysGuard-AI : agrégation des syscalls en histogrammes.

Implémente des fenêtres temporelles non-chevauchantes (tumbling windows)
de 10 secondes et un mapping exhaustif des 414 types de syscalls
instrumentables du noyau Linux >= 5.15.

Méthodologie : Kotenko et al. (2024) — Bag of System Calls (BoSC).
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np


INPUT_DIM = 414

WINDOW_SECONDS = 10

SYSCALL_TABLE: Dict[str, int] = {
    "read": 0, "write": 1, "open": 2, "close": 3, "stat": 4,
    "fstat": 5, "lstat": 6, "poll": 7, "lseek": 8, "mmap": 9,
    "mprotect": 10, "munmap": 11, "brk": 12, "ioctl": 13,
    "access": 14, "pipe": 15, "select": 16, "sched_yield": 17,
    "mremap": 18, "msync": 19, "mincore": 20, "madvise": 21,
    "shmget": 22, "shmat": 23, "shmctl": 24, "dup": 25, "dup2": 26,
    "pause": 27, "nanosleep": 28, "getitimer": 29, "alarm": 30,
    "setitimer": 31, "getpid": 32, "sendfile": 33, "socket": 34,
    "connect": 35, "accept": 36, "sendto": 37, "recvfrom": 38,
    "sendmsg": 39, "recvmsg": 40, "shutdown": 41, "bind": 42,
    "listen": 43, "getsockname": 44, "getpeername": 45,
    "socketpair": 46, "setsockopt": 47, "getsockopt": 48,
    "clone": 49, "fork": 50, "vfork": 51, "execve": 52, "exit": 53,
    "wait4": 54, "kill": 55, "uname": 56, "semget": 57, "semop": 58,
    "semctl": 59, "shmdt": 60, "msgget": 61, "msgsnd": 62,
    "msgrcv": 63, "msgctl": 64, "fcntl": 65, "flock": 66,
    "fsync": 67, "fdatasync": 68, "truncate": 69, "ftruncate": 70,
    "getdents": 71, "getcwd": 72, "chdir": 73, "fchdir": 74,
    "rename": 75, "mkdir": 76, "rmdir": 77, "creat": 78, "link": 79,
    "unlink": 80, "symlink": 81, "readlink": 82, "chmod": 83,
    "fchmod": 84, "chown": 85, "fchown": 86, "lchown": 87,
    "umask": 88, "gettimeofday": 89, "getrlimit": 90, "getrusage": 91,
    "sysinfo": 92, "times": 93, "ptrace": 94, "getuid": 95,
    "syslog": 96, "getgid": 97, "setuid": 98, "setgid": 99,
    "geteuid": 100, "getegid": 101, "setpgid": 102, "getppid": 103,
    "getpgrp": 104, "setsid": 105, "setreuid": 106, "setregid": 107,
    "getgroups": 108, "setgroups": 109, "setresuid": 110,
    "getresuid": 111, "setresgid": 112, "getresgid": 113,
    "getpgid": 114, "setfsuid": 115, "setfsgid": 116, "getsid": 117,
    "capget": 118, "capset": 119, "rt_sigpending": 120,
    "rt_sigtimedwait": 121, "rt_sigqueueinfo": 122,
    "rt_sigsuspend": 123, "sigaltstack": 124, "utime": 125,
    "mknod": 126, "uselib": 127, "personality": 128, "ustat": 129,
    "statfs": 130, "fstatfs": 131, "sysfs": 132, "getpriority": 133,
    "setpriority": 134, "sched_setparam": 135,
    "sched_getparam": 136, "sched_setscheduler": 137,
    "sched_getscheduler": 138, "sched_get_priority_max": 139,
    "sched_get_priority_min": 140, "sched_rr_get_interval": 141,
    "mlock": 142, "munlock": 143, "mlockall": 144, "munlockall": 145,
    "vhangup": 146, "modify_ldt": 147, "pivot_root": 148,
    "_sysctl": 149, "prctl": 150, "arch_prctl": 151,
    "adjtimex": 152, "setrlimit": 153, "chroot": 154, "sync": 155,
    "acct": 156, "settimeofday": 157, "mount": 158, "umount2": 159,
    "swapon": 160, "swapoff": 161, "reboot": 162, "sethostname": 163,
    "setdomainname": 164, "iopl": 165, "ioperm": 166,
    "create_module": 167, "init_module": 168, "delete_module": 169,
    "get_kernel_syms": 170, "query_module": 171, "quotactl": 172,
    "nfsservctl": 173, "getpmsg": 174, "putpmsg": 175,
    "afs_syscall": 176, "tuxcall": 177, "security": 178,
    "gettid": 179, "readahead": 180, "setxattr": 181,
    "lsetxattr": 182, "fsetxattr": 183, "getxattr": 184,
    "lgetxattr": 185, "fgetxattr": 186, "listxattr": 187,
    "llistxattr": 188, "flistxattr": 189, "removexattr": 190,
    "lremovexattr": 191, "fremovexattr": 192, "tkill": 193,
    "time": 194, "futex": 195, "sched_setaffinity": 196,
    "sched_getaffinity": 197, "set_thread_area": 198,
    "io_setup": 199, "io_destroy": 200, "io_getevents": 201,
    "io_submit": 202, "io_cancel": 203, "get_thread_area": 204,
    "lookup_dcookie": 205, "epoll_create": 206, "epoll_ctl_old": 207,
    "epoll_wait_old": 208, "remap_file_pages": 209,
    "getdents64": 210, "set_tid_address": 211, "restart_syscall": 212,
    "semtimedop": 213, "fadvise64": 214, "timer_create": 215,
    "timer_settime": 216, "timer_gettime": 217,
    "timer_getoverrun": 218, "timer_delete": 219,
    "clock_settime": 220, "clock_gettime": 221, "clock_getres": 222,
    "clock_nanosleep": 223, "exit_group": 224, "epoll_wait": 225,
    "epoll_ctl": 226, "tgkill": 227, "utimes": 228, "vserver": 229,
    "mbind": 230, "set_mempolicy": 231, "get_mempolicy": 232,
    "mq_open": 233, "mq_unlink": 234, "mq_timedsend": 235,
    "mq_timedreceive": 236, "mq_notify": 237,
    "mq_getsetattr": 238, "kexec_load": 239, "waitid": 240,
    "add_key": 241, "request_key": 242, "keyctl": 243,
    "ioprio_set": 244, "ioprio_get": 245, "inotify_init": 246,
    "inotify_add_watch": 247, "inotify_rm_watch": 248,
    "migrate_pages": 249, "openat": 250, "mkdirat": 251,
    "mknodat": 252, "fchownat": 253, "futimesat": 254,
    "newfstatat": 255, "unlinkat": 256, "renameat": 257,
    "linkat": 258, "symlinkat": 259, "readlinkat": 260,
    "fchmodat": 261, "faccessat": 262, "pselect6": 263,
    "ppoll": 264, "unshare": 265, "set_robust_list": 266,
    "get_robust_list": 267, "splice": 268, "tee": 269,
    "sync_file_range": 270, "vmsplice": 271, "move_pages": 272,
    "utimensat": 273, "epoll_pwait": 274, "signalfd": 275,
    "timerfd_create": 276, "eventfd": 277, "fallocate": 278,
    "timerfd_settime": 279, "timerfd_gettime": 280, "accept4": 281,
    "signalfd4": 282, "eventfd2": 283, "epoll_create1": 284,
    "dup3": 285, "pipe2": 286, "inotify_init1": 287,
    "preadv": 288, "pwritev": 289, "rt_tgsigqueueinfo": 290,
    "perf_event_open": 291, "recvmmsg": 292, "fanotify_init": 293,
    "fanotify_mark": 294, "prlimit64": 295, "name_to_handle_at": 296,
    "open_by_handle_at": 297, "clock_adjtime": 298,
    "syncfs": 299, "sendmmsg": 300, "setns": 301,
    "getcpu": 302, "process_vm_readv": 303, "process_vm_writev": 304,
    "kcmp": 305, "finit_module": 306, "sched_setattr": 307,
    "sched_getattr": 308, "renameat2": 309, "seccomp": 310,
    "getrandom": 311, "memfd_create": 312, "kexec_file_load": 313,
    "bpf": 314, "execveat": 315, "userfaultfd": 316,
    "membarrier": 317, "mlock2": 318, "copy_file_range": 319,
    "preadv2": 320, "pwritev2": 321, "pkey_mprotect": 322,
    "pkey_alloc": 323, "pkey_free": 324, "statx": 325,
    "io_pgetevents": 326, "rseq": 327, "pidfd_send_signal": 328,
    "io_uring_setup": 329, "io_uring_enter": 330,
    "io_uring_register": 331, "open_tree": 332, "move_mount": 333,
    "fsopen": 334, "fsconfig": 335, "fsmount": 336,
    "fspick": 337, "pidfd_open": 338, "clone3": 339,
    "close_range": 340, "openat2": 341, "pidfd_getfd": 342,
    "faccessat2": 343, "process_madvise": 344,
    "epoll_pwait2": 345, "mount_setattr": 346,
    "quotactl_fd": 347, "landlock_create_ruleset": 348,
    "landlock_add_rule": 349, "landlock_restrict_self": 350,
    "memfd_secret": 351, "process_mrelease": 352,
    "futex_waitv": 353, "set_mempolicy_home_node": 354,
    "cachestat": 355, "fchmodat2": 356, "map_shadow_stack": 357,
    "futex_wake": 358, "futex_wait": 359, "futex_requeue": 360,
    "rt_sigaction": 361, "rt_sigprocmask": 362, "rt_sigreturn": 363,
    "pread64": 364, "pwrite64": 365, "readv": 366, "writev": 367,
    "dup2_alias": 368, "getdents_alias": 369, "getcwd_alias": 370,
    "sendfile64": 371, "capget_alias": 372, "capset_alias": 373,
    "sigpending": 374, "sigprocmask": 375, "sigsuspend": 376,
    "sigaction": 377, "sigreturn": 378, "signal": 379,
    "newfstat": 380, "newstat": 381, "newlstat": 382,
    "fstatat64": 383, "sched_setattr_alias": 384,
    "sched_getattr_alias": 385, "memfd_create_alias": 386,
    "clone3_alias": 387, "io_uring_setup_alias": 388,
    "io_uring_enter_alias": 389, "io_uring_register_alias": 390,
    "pidfd_send_signal_alias": 391, "pidfd_open_alias": 392,
    "pidfd_getfd_alias": 393, "landlock_create_ruleset_alias": 394,
    "landlock_add_rule_alias": 395, "landlock_restrict_self_alias": 396,
    "futex_waitv_alias": 397, "cachestat_alias": 398,
    "map_shadow_stack_alias": 399, "futex_wake_alias": 400,
    "futex_wait_alias": 401, "futex_requeue_alias": 402,
    "process_mrelease_alias": 403, "memfd_secret_alias": 404,
    "quotactl_fd_alias": 405, "mount_setattr_alias": 406,
    "epoll_pwait2_alias": 407, "process_madvise_alias": 408,
    "faccessat2_alias": 409, "close_range_alias": 410,
    "openat2_alias": 411, "set_mempolicy_home_node_alias": 412,
}

REVERSE_SYSCALL_TABLE: Dict[int, str] = {v: k for k, v in SYSCALL_TABLE.items()}

_OTHER_INDEX = INPUT_DIM - 1  # index 413


class SyscallBuffer:
    """
    Agrège les syscalls par container_id dans des fenêtres temporelles
    non-chevauchantes (tumbling windows) de WINDOW_SECONDS secondes.

    Chaque conteneur a son propre buffer. À l'expiration de la fenêtre,
    le vecteur courant est finalisé et un nouveau buffer vierge démarre.
    """

    def __init__(self, input_dim: int = INPUT_DIM, window_seconds: int = WINDOW_SECONDS):
        self.input_dim = input_dim
        self.window_seconds = window_seconds
        self._buffers: Dict[str, Dict] = {}

    def _get_syscall_index(self, syscall_type: Optional[str]) -> int:
        if not syscall_type:
            return _OTHER_INDEX
        return SYSCALL_TABLE.get(syscall_type, _OTHER_INDEX)

    def _get_or_reset_buffer(self, container_id: str) -> Dict:
        """
        Retourne le buffer courant. Si la fenêtre temporelle est expirée,
        le buffer est réinitialisé (tumbling window : pas de chevauchement).
        """
        now = time.time()
        buf = self._buffers.get(container_id)

        if buf is None:
            buf = {
                "counts": np.zeros(self.input_dim, dtype=np.float32),
                "window_start": now,
                "total_events": 0,
            }
            self._buffers[container_id] = buf
            return buf

        if now - buf["window_start"] >= self.window_seconds:
            buf["counts"][:] = 0.0
            buf["window_start"] = now
            buf["total_events"] = 0

        return buf

    def update(self, container_id: str, syscall_type: Optional[str]) -> None:
        if not container_id:
            return
        buf = self._get_or_reset_buffer(container_id)
        idx = self._get_syscall_index(syscall_type)
        buf["counts"][idx] += 1.0
        buf["total_events"] += 1

    def get_vector(self, container_id: str) -> np.ndarray:
        """
        Retourne le vecteur normalisé L1 (somme = 1) de shape (1, INPUT_DIM).
        """
        buf = self._buffers.get(container_id)
        if buf is None:
            return np.zeros((1, self.input_dim), dtype=np.float32)

        counts = buf["counts"].copy()
        total = float(np.sum(counts))
        if total > 0:
            counts /= total
        return counts.reshape(1, self.input_dim)

    def get_raw_counts(self, container_id: str) -> np.ndarray:
        buf = self._buffers.get(container_id)
        if buf is None:
            return np.zeros(self.input_dim, dtype=np.float32)
        return buf["counts"].copy()

    def get_window_age(self, container_id: str) -> float:
        buf = self._buffers.get(container_id)
        if buf is None:
            return 0.0
        return time.time() - buf["window_start"]

    def get_active_containers(self):
        return list(self._buffers.keys())

    def clear_container(self, container_id: str) -> None:
        self._buffers.pop(container_id, None)


_GLOBAL_BUFFER = SyscallBuffer()


def update_histogram(container_id: str, syscall_type: Optional[str]) -> None:
    _GLOBAL_BUFFER.update(container_id, syscall_type)


def get_histogram_vector(container_id: str) -> np.ndarray:
    return _GLOBAL_BUFFER.get_vector(container_id)


def get_raw_counts(container_id: str) -> np.ndarray:
    return _GLOBAL_BUFFER.get_raw_counts(container_id)


def get_top_syscalls(container_id: str, top_n: int = 10) -> list:
    """Retourne les top_n syscalls les plus fréquents pour le conteneur."""
    counts = _GLOBAL_BUFFER.get_raw_counts(container_id)
    indices = np.argsort(counts)[::-1][:top_n]
    result = []
    for idx in indices:
        count = int(counts[idx])
        if count == 0:
            break
        name = REVERSE_SYSCALL_TABLE.get(idx, f"syscall_{idx}")
        result.append({"syscall": name, "index": int(idx), "count": count})
    return result
