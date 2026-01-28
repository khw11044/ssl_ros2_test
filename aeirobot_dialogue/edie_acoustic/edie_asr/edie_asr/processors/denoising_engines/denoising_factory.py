
from .deepfilter_denoising import DeepfilterDenoising
from .rnnoise_denoising import RnnoiseDenoising


class DenoisingFactory:
    @staticmethod
    def create_denoising(node):
        """
        Denoising 인스턴스 생성

        node.config에서 'denoising' 섹션을 읽어 Denoising 타입에 맞는 인스턴스를 생성합니다.
        config에 없는 인자는 각 Denoising 클래스의 기본값이 사용됩니다.

        Args:
            node (Node): ROS2 노드 (node.config에서 설정을 읽음)

        Returns:
            BaseDenoising: Denoising 인스턴스
        """
        config = node.config
        denoising_config = config.get('denoising', {}).copy()
        denoising_type = denoising_config.pop('type', 'deepfilter')  # type 추출 및 제거

        if denoising_type.lower() == 'deepfilter':
            return DeepfilterDenoising(**denoising_config)
        elif denoising_type.lower() == 'rnnoise':
            return RnnoiseDenoising(**denoising_config)
        else:
            raise ValueError(f'Unknown denoising type: {denoising_type}')
