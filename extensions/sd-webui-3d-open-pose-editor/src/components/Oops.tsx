import i18n from '../i18n'
import { ShowDialog } from './Dialog'

export function Oops(error: any) {
    const stack = error?.stack
    const text = `${i18n.t('Something went wrong!')}\n${stack || error}`
    ShowDialog({
        title: i18n.t('Oops...') ?? '',
        description: text,
        children: (
            <a href="https://github.com/nonnonstop/sd-webui-3d-open-pose-editor/issues/new/choose">
                {i18n.t(
                    'If the problem persists, please click here to ask a question.'
                )}
            </a>
        ),
        button: i18n.t('Close') ?? '',
    })
}
