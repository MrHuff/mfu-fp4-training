#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

from low_bits_training import evaluation
import sys


def main():
    parser = evaluation.get_shared_parser()
    args, unknown_args = parser.parse_known_args()
    return_code = evaluation.EVALUATION_METHODS[args.method].run(
        shared_args=args,
        own_args=args,
        unknown_args=unknown_args,
    )
    if return_code is not None:
        sys.exit(return_code)


if __name__ == "__main__":
    main()
