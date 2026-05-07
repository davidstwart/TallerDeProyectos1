from domain.models.tema_model import (
    Tema
)

class TemaUseCase:

    def __init__(
        self,
        tema_repository
    ):

        self.tema_repository = (
            tema_repository
        )

    # =====================================
    # CREAR
    # =====================================

    def create_tema(
        self,
        data
    ):

        tema = Tema(
            id_tema=None,
            nombre=data.nombre,
            descripcion=data.descripcion,
            activo=True
        )

        return self.tema_repository.save(
            tema
        )

    # =====================================
    # LISTAR
    # =====================================

    def get_temas(self):

        return (
            self.tema_repository.get_all()
        )