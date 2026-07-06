from dependency_injector import containers, providers
from backend.application_context import ApplicationContext


class Container(containers.DeclarativeContainer):
    context = providers.Singleton(ApplicationContext)


container = Container()
