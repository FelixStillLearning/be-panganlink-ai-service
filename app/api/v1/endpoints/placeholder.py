from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_placeholder() -> dict[str, str]:
    return {"message": "placeholder"}
